import base64
import contextlib
import io
import json
import struct
import tempfile
import threading
import unittest
import urllib.error
from pathlib import Path

from agent.jev import JEV_MODEL, JEV_TIMEOUT_SECONDS, JEV_URL
from agent.server import (
    JSON_ERROR,
    MAX_BODY_BYTES,
    MAX_VISIT_BODY_BYTES,
    Handler,
    body_length_error,
    listen,
)
from agent.settings import settings_from
from agent.visit import (
    JPEG_ERROR,
    MAX_SIDE,
    MIN_SIDE,
    SIZE_ERROR,
    VISIT_BUDGET_SECONDS,
    jpeg_size,
    parse_visit_multipart,
    validate_visit_jpeg,
    verify_response,
)
from agent.visit_embedding import (
    MODEL_ID,
    VECTOR_DIM,
    ReferenceCatalog,
    StoredView,
    embedding_signal,
    load_catalog,
    nearest_similarity,
)
from agent.visit_jev import (
    IMAGE_HEADER,
    JEV_VISIT_TIMEOUT_SECONDS,
    LOOKS_LIKE_SCREEN,
    MATCHES_PLACE,
    ask_visit,
    read_visit_answers,
)

NO_JEV = settings_from({"JEV_API_KEY": ""})
WITH_JEV = settings_from({"JEV_API_KEY": "test-key"})
PLACE = {
    "placeId": "plaza-de-mayo",
    "name": "Plaza de Mayo",
    "category": "plaza",
    "city": "Buenos Aires",
    "country": "Argentina",
    "xp": 40,
    "stamp": "oro",
}
BOUNDARY = "----visitBoundary7"


def jpeg_with_size(width: int, height: int, marker: int = 0xC0) -> bytes:
    payload = struct.pack(">BHHB3B", 8, height, width, 1, 1, 0x11, 0)
    segment = bytes((0xFF, marker)) + struct.pack(">H", 2 + len(payload)) + payload
    app0 = bytes.fromhex("ffe000104a46494600010100000100010000")
    return b"\xff\xd8" + app0 + segment + b"\xff\xd9"


def form(payload, photo, boundary=BOUNDARY, visit_type="image/jpeg", request_type="application/json", preamble=b"", reverse=False, extra=None):
    request = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")

    def part(name, content_type, body, filename=None):
        disposition = f'Content-Disposition: form-data; name="{name}"'
        if filename:
            disposition += f'; filename="{filename}"'
        head = f"--{boundary}\r\n{disposition}\r\n"
        if content_type is not None:
            head += f"Content-Type: {content_type}\r\n"
        return head.encode("utf-8") + b"\r\n" + body

    chunks = [part("request", request_type, request), part("visit", visit_type, photo, "visita.jpg")]
    if reverse:
        chunks.reverse()
    if extra is not None:
        chunks.append(part("note", "text/plain", extra))
    body = preamble + b"\r\n".join(chunks) + f"\r\n--{boundary}--\r\n".encode("utf-8")
    return f'multipart/form-data; boundary="{boundary}"', body


class FakeAPI:
    def __init__(self, payload: bytes, status: int = 200):
        self.payload = payload
        self.status = status
        self.request = None
        self.timeout = None

    def open(self, request, timeout=None):
        self.request = request
        self.timeout = timeout
        if self.status != 200:
            raise urllib.error.HTTPError(request.full_url, self.status, "error", None, None)
        return FakeBody(self.payload)


class FakeBody:
    def __init__(self, payload: bytes):
        self.payload = payload

    def read(self, size: int) -> bytes:
        return self.payload[:size]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class MultipartTests(unittest.TestCase):
    def test_parser_reads_json_and_the_jpeg_bytes(self):
        photo = jpeg_with_size(320, 480) + b"\r\n--not-the-boundary\r\n\xff\x00"
        content_type, body = form(PLACE, photo, preamble=b"preambulo\r\n")
        parsed = parse_visit_multipart(content_type, body)
        self.assertEqual(json.loads(parsed.request), PLACE)
        self.assertEqual(parsed.photo, photo)

    def test_parts_can_arrive_with_the_photo_first(self):
        photo = jpeg_with_size(280, 1568)
        content_type, body = form(PLACE, photo, reverse=True)
        parsed = parse_visit_multipart(content_type, body)
        self.assertEqual(parsed.photo, photo)

    def test_missing_photo_extra_part_and_non_json_are_rejected(self):
        photo = jpeg_with_size(280, 280)
        content_type, body = form(PLACE, photo)
        request_only = body.split(f"\r\n--{BOUNDARY}\r\n".encode(), 2)[0] + f"\r\n--{BOUNDARY}--\r\n".encode()
        # The split above is brittle if the photo contains the boundary. This photo does not.
        with self.assertRaises(Exception) as missing:
            parse_visit_multipart(content_type, request_only)
        self.assertIn("foto", str(missing.exception).lower())

        _type, extra = form(PLACE, photo, extra=b"nota")
        with self.assertRaises(Exception) as too_many:
            parse_visit_multipart(_type, extra)
        self.assertIn("partes", str(too_many.exception).lower())

        _type, bad = form(b"no-es-json", photo)
        with self.assertRaises(Exception) as invalid:
            from agent.visit import parse_visit_fields

            parse_visit_fields(parse_visit_multipart(_type, bad).request)
        self.assertIn("JSON", str(invalid.exception))


class JpegTests(unittest.TestCase):
    def test_sides_outside_280_to_1568_are_rejected(self):
        self.assertEqual((MIN_SIDE, MAX_SIDE), (280, 1568))
        self.assertIsNone(validate_visit_jpeg(jpeg_with_size(280, 1568)))
        self.assertIsNone(validate_visit_jpeg(jpeg_with_size(1568, 280, marker=0xC2)))
        self.assertEqual(validate_visit_jpeg(jpeg_with_size(279, 400)), SIZE_ERROR)
        self.assertEqual(validate_visit_jpeg(jpeg_with_size(400, 1569)), SIZE_ERROR)
        self.assertEqual(validate_visit_jpeg(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16), JPEG_ERROR)
        self.assertIsNone(jpeg_size(b"\xff\xd8\xff\xd9"))
        self.assertEqual(jpeg_size(jpeg_with_size(300, 400)), (300, 400))


class VisitResponseTests(unittest.TestCase):
    def test_valid_visit_returns_signals_without_stamp_or_xp(self):
        content_type, body = form(PLACE, jpeg_with_size(640, 480))
        status, payload = verify_response(content_type, body, NO_JEV, catalog=ReferenceCatalog())
        self.assertEqual(status, 200)
        self.assertEqual(list(payload), ["embedding", "jev"])
        self.assertEqual(payload["embedding"], {"available": False})
        self.assertEqual(payload["jev"], {"available": False})
        self.assertNotIn("xp", payload)
        self.assertNotIn("stamp", payload)

    def test_bad_file_is_rejected_before_jev(self):
        def fail_if_called(*args, **kwargs):
            raise AssertionError("jev")

        content_type, body = form(PLACE, jpeg_with_size(100, 100))
        status, payload = verify_response(
            content_type,
            body,
            WITH_JEV,
            catalog=ReferenceCatalog(),
            ask=fail_if_called,
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload["message"], SIZE_ERROR)

        _type, png = form(PLACE, b"\x89PNG\r\n\x1a\n", visit_type="image/png")
        status, payload = verify_response(_type, png, NO_JEV, catalog=ReferenceCatalog(), ask=fail_if_called)
        self.assertEqual(status, 400)
        self.assertEqual(payload["message"], JPEG_ERROR)

    def test_missing_fields_do_not_call_jev(self):
        called = []
        content_type, body = form({"placeId": "plaza-de-mayo", "name": "Plaza de Mayo"}, jpeg_with_size(280, 280))
        status, payload = verify_response(
            content_type,
            body,
            WITH_JEV,
            catalog=ReferenceCatalog(),
            ask=lambda *args, **kwargs: called.append(True),
        )
        self.assertEqual(status, 400)
        self.assertIn("categoría", payload["message"])
        self.assertIn("ciudad", payload["message"])
        self.assertIn("país", payload["message"])
        self.assertEqual(called, [])


class JevVisitTests(unittest.TestCase):
    def test_simulated_api_returns_both_probabilities(self):
        photo = jpeg_with_size(300, 400) + b"\r\n--sigue\r\n"
        api = FakeAPI(
            json.dumps(
                {
                    "model": "jev-1",
                    "answers": {
                        MATCHES_PLACE: {"type": "noul", "noul": 0.91},
                        LOOKS_LIKE_SCREEN: {"type": "noul", "noul": 0.08},
                    },
                }
            ).encode("utf-8")
        )
        result = ask_visit(
            photo,
            place_id="plaza-de-mayo",
            name="Plaza de Mayo",
            category="plaza",
            city="Buenos Aires",
            country="Argentina",
            settings=WITH_JEV,
            timeout=JEV_VISIT_TIMEOUT_SECONDS,
            opener=api,
        )
        self.assertEqual(result, (0.91, 0.08))
        self.assertEqual(JEV_TIMEOUT_SECONDS, 1.5)
        self.assertEqual(api.timeout, 10.0)
        self.assertLess(JEV_VISIT_TIMEOUT_SECONDS, VISIT_BUDGET_SECONDS)
        self.assertEqual(api.request.full_url, JEV_URL)
        self.assertEqual(api.request.get_method(), "POST")
        self.assertEqual(api.request.headers[IMAGE_HEADER], "true")
        self.assertEqual(api.request.headers["Authorization"], "Bearer test-key")

        content_type = api.request.headers["Content-type"]
        parsed = parse_visit_multipart(content_type, api.request.data)
        sent = json.loads(parsed.request)
        self.assertEqual(parsed.photo, photo)
        self.assertEqual(sent["model"], JEV_MODEL)
        ref = sent["state"]["photo"]
        self.assertEqual(sent["assets"][ref], {"type": "image", "part": "visit"})
        self.assertEqual(sent["state"]["place_name"], "Plaza de Mayo")
        self.assertEqual(sent["state"]["place_category"], "plaza")
        self.assertEqual(sent["state"]["city"], "Buenos Aires")
        self.assertNotIn(ref, json.dumps(sent["questions"]))
        self.assertNotIn("data_base64", parsed.request.decode("utf-8"))
        questions = sent["questions"]
        self.assertEqual(set(questions), {MATCHES_PLACE, LOOKS_LIKE_SCREEN})
        for question in questions.values():
            self.assertEqual(question["type"], "noul")
            self.assertIn("photo", question["instructions"])
            self.assertIn("true", question["criteria"])
            self.assertIn("false", question["criteria"])
        self.assertIn("place_name", questions[MATCHES_PLACE]["instructions"])
        self.assertIn("pantalla", questions[LOOKS_LIKE_SCREEN]["instructions"])
        self.assertIn("impres", questions[LOOKS_LIKE_SCREEN]["instructions"])

        content_type, body = form(PLACE, photo)
        seen = {}

        def ask(photo_bytes, timeout, **kwargs):
            seen["timeout"] = timeout
            seen["photo"] = photo_bytes
            return result

        status, payload = verify_response(content_type, body, WITH_JEV, catalog=ReferenceCatalog(), ask=ask)
        self.assertEqual(status, 200)
        self.assertEqual(payload["embedding"], {"available": False})
        self.assertEqual(payload["jev"], {"available": True, "matchesPlace": 0.91, "looksLikeScreen": 0.08})
        self.assertGreater(seen["timeout"], JEV_TIMEOUT_SECONDS)
        self.assertLessEqual(seen["timeout"], JEV_VISIT_TIMEOUT_SECONDS)
        self.assertEqual(seen["photo"], photo)

    def test_failed_api_does_not_invent_a_probability_or_log_the_photo(self):
        photo = jpeg_with_size(280, 280)
        api = FakeAPI(b"", status=500)
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            result = ask_visit(
                photo,
                place_id="plaza-de-mayo",
                name="Plaza de Mayo",
                category="plaza",
                city="Buenos Aires",
                country="Argentina",
                settings=WITH_JEV,
                timeout=10,
                opener=api,
            )
        self.assertIsNone(result)
        self.assertEqual(stderr.getvalue(), "jev visita respondió 500\n")
        self.assertNotIn(base64.b64encode(photo).decode("ascii"), stderr.getvalue())

        incomplete = read_visit_answers({"answers": {MATCHES_PLACE: {"type": "noul", "noul": 0.91}}})
        self.assertIsNone(incomplete)
        self.assertIsNone(read_visit_answers({"answers": {MATCHES_PLACE: {"type": "noul", "noul": True}, LOOKS_LIKE_SCREEN: {"type": "noul", "noul": 0.08}}}))
        self.assertIsNone(read_visit_answers({"answers": {MATCHES_PLACE: {"type": "noul", "noul": 1.2}, LOOKS_LIKE_SCREEN: {"type": "noul", "noul": 0.08}}}))
        self.assertEqual(read_visit_answers({"answers": {MATCHES_PLACE: {"noul": 0}, LOOKS_LIKE_SCREEN: {"noul": 1}}}), (0.0, 1.0))

        content_type, body = form(PLACE, photo)
        status, payload = verify_response(
            content_type,
            body,
            WITH_JEV,
            catalog=ReferenceCatalog(),
            ask=lambda *args, **kwargs: None,
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["jev"], {"available": False})
        self.assertNotIn("matchesPlace", payload["jev"])
        self.assertNotIn("looksLikeScreen", payload["jev"])


class EmbeddingTests(unittest.TestCase):
    def test_without_references_the_photo_is_not_scored(self):
        calls = []
        signal = embedding_signal("plaza-de-mayo", b"\xff\xd8foto", ReferenceCatalog(), lambda photo: calls.append(photo))
        self.assertEqual(signal, {"available": False})
        self.assertEqual(calls, [])
        held = ReferenceCatalog((StoredView("plaza-de-mayo", MODEL_ID, "norte", (1.0, 0.0)),))
        self.assertEqual(embedding_signal("plaza-de-mayo", b"\xff\xd8foto", held), {"available": False})
        self.assertTrue(MODEL_ID.startswith("dinov2"))
        self.assertNotIn("clip", MODEL_ID)
        self.assertEqual(VECTOR_DIM, 768)

    def test_closest_view_of_the_same_model_wins(self):
        far = StoredView("plaza-de-mayo", MODEL_ID, "lejos", (1.0, 0.0))
        near = StoredView("plaza-de-mayo", MODEL_ID, "cerca", (0.0, 1.0))
        other_model = StoredView("plaza-de-mayo", "clip-vit-b32", "clip", (0.0, 1.0))
        other_place = StoredView("obelisco", MODEL_ID, "ajeno", (0.0, 1.0))
        catalog = ReferenceCatalog((far, near, other_model, other_place))
        self.assertEqual(
            [view.view_id for view in catalog.for_place("plaza-de-mayo", MODEL_ID)],
            ["lejos", "cerca"],
        )
        signal = embedding_signal("plaza-de-mayo", b"\xff\xd8", catalog, lambda _photo: (0.0, 1.0))
        self.assertEqual(signal["available"], True)
        self.assertAlmostEqual(signal["nearest"], 1.0)
        ignored = embedding_signal(
            "plaza-de-mayo",
            b"\xff\xd8",
            ReferenceCatalog((far, other_model)),
            lambda _photo: (0.0, 1.0),
        )
        self.assertAlmostEqual(ignored["nearest"], 0.0)
        self.assertIsNone(nearest_similarity(MODEL_ID, (0.0, 1.0), (other_model,)))

    def test_stored_views_keep_the_model_id(self):
        good = [0.0] * VECTOR_DIM
        good[0] = 1.0
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "norte.json").write_text(
                json.dumps({"placeId": "plaza-de-mayo", "modelId": MODEL_ID, "viewId": "norte", "vector": good}),
                encoding="utf-8",
            )
            (root / "sur.json").write_text(
                json.dumps(
                    {
                        "placeId": "plaza-de-mayo",
                        "modelId": MODEL_ID,
                        "viewId": "sur",
                        "vector": [1.0] + [0.0] * (VECTOR_DIM - 1),
                    }
                ),
                encoding="utf-8",
            )
            (root / "clip.json").write_text(
                json.dumps({"placeId": "plaza-de-mayo", "modelId": "clip-vit-b32", "viewId": "clip", "vector": good}),
                encoding="utf-8",
            )
            (root / "corto.json").write_text(
                json.dumps({"placeId": "plaza-de-mayo", "modelId": MODEL_ID, "viewId": "corto", "vector": [1, 2, 3]}),
                encoding="utf-8",
            )
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                catalog = load_catalog(root)
            views = catalog.for_place("plaza-de-mayo", MODEL_ID)
            self.assertEqual([view.view_id for view in views], ["norte", "sur"])
            self.assertTrue(all(view.model_id == MODEL_ID and len(view.vector) == VECTOR_DIM for view in views))
            self.assertIn("corto.json", stderr.getvalue())
            self.assertNotIn("1, 2, 3", stderr.getvalue())
            self.assertEqual(load_catalog(root / "ausente").views, ())


class BodyLimitTests(unittest.TestCase):
    def test_context_stays_at_64_kib_and_visit_has_its_own_limit(self):
        self.assertEqual(MAX_BODY_BYTES, 65_536)
        self.assertEqual(MAX_VISIT_BODY_BYTES, 8 * 1024 * 1024)
        self.assertEqual(body_length_error(None, None, length_count=0), JSON_ERROR)
        self.assertEqual(body_length_error("10", "chunked"), JSON_ERROR)
        self.assertIn("demasiado grande", body_length_error("70000", None) or "")
        self.assertIsNone(body_length_error("70000", None, max_bytes=MAX_VISIT_BODY_BYTES))
        self.assertIsNone(body_length_error(str(MAX_VISIT_BODY_BYTES), None, max_bytes=MAX_VISIT_BODY_BYTES))
        self.assertIn(
            "demasiado grande",
            body_length_error(str(MAX_VISIT_BODY_BYTES + 1), None, max_bytes=MAX_VISIT_BODY_BYTES) or "",
        )


class HttpRouteTests(unittest.TestCase):
    def test_verify_route_is_separate_from_context(self):
        Handler.settings = NO_JEV
        server = listen("127.0.0.1", 0)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            content_type, raw = form(PLACE, jpeg_with_size(280, 280))
            status, payload = request(port, "POST", "/visits/verify", raw, {"Content-Type": content_type})
            self.assertEqual(status, 200)
            self.assertEqual(payload["embedding"], {"available": False})
            self.assertEqual(payload["jev"], {"available": False})

            status, payload = request(port, "GET", "/visits/verify", b"", {})
            self.assertEqual(status, 405)

            oversized = b"{" * 70_000
            status, payload = request(
                port,
                "POST",
                "/context",
                oversized,
                {"Content-Type": "application/json"},
            )
            self.assertEqual(status, 400)
            self.assertIn("demasiado grande", payload["message"])

            context = json.dumps(
                {
                    "location": {"latitude": -34.6037, "longitude": -58.3816},
                    "pointsOfInterest": [
                        {
                            "name": "Plaza de Mayo",
                            "latitude": -34.6083,
                            "longitude": -58.3722,
                            "category": "plaza",
                            "distanceMeters": 120,
                        }
                    ],
                }
            ).encode("utf-8")
            status, payload = request(port, "POST", "/context", context, {"Content-Type": "application/json"})
            self.assertEqual(status, 200)
            self.assertIn("Plaza de Mayo", payload["reply"])
        finally:
            server.shutdown()
            server.server_close()


def request(port: int, method: str, path: str, body: bytes, headers: dict[str, str]):
    import http.client

    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        raw = response.read()
        return response.status, json.loads(raw.decode("utf-8"))
    finally:
        connection.close()


if __name__ == "__main__":
    unittest.main()
