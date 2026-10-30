import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.cluster import cluster_interests, rank_place, zone_label
from agent.contract import Place, parse_context
from agent.deadline import Deadline
from agent.graph import build_guide, build_reply, compile_graph
from agent.web_context import WebPassage, match_score, pick_page
from agent.history import SHARED_MEMORY, MentionMemory
from agent.jev import CRITERIA_FALSE, CRITERIA_TRUE, build_request, simulate_noul
from agent.retrieve import KNOWLEDGE_DIR, load_index, read_knowledge, search_history
from agent.server import body_length_error, response_for
from agent.settings import settings_from
from agent.situation import analyze_situation
from agent.voice import ensure_place_name, mentions_place

DISABLED = settings_from({"MODEL_API_KEY": "", "MODEL_BASE_URL": "https://api.openai.com/v1", "MODEL_NAME": "gpt-4o-mini"})


def context_body(**overrides):
    body = {
        "userId": "11111111-1111-1111-1111-111111111111",
        "location": {"latitude": -34.6037, "longitude": -58.3816},
        "pointsOfInterest": [
            {
                "id": "plaza",
                "name": "Plaza de Mayo",
                "latitude": -34.6083,
                "longitude": -58.3722,
                "category": "plaza",
                "distanceMeters": 120,
            }
        ],
    }
    body.update(overrides)
    return body


class ContractTests(unittest.TestCase):
    def setUp(self) -> None:
        SHARED_MEMORY.clear()

    def test_missing_location_is_400(self):
        status, body = response_for(json.dumps(context_body(location=None)).encode(), DISABLED)
        self.assertEqual(status, 400)
        self.assertIn("Falta la ubicación", body["message"])

    def test_places_must_be_a_list(self):
        status, body = response_for(
            json.dumps(context_body(pointsOfInterest={"name": "Plaza"})).encode(),
            DISABLED,
        )
        self.assertEqual(status, 400)
        self.assertEqual(body["message"], "Los lugares tienen que ser una lista")

    def test_invalid_coordinates_are_400(self):
        status, body = response_for(
            json.dumps(context_body(location={"latitude": 120, "longitude": 10})).encode(),
            DISABLED,
        )
        self.assertEqual(status, 400)
        self.assertEqual(body["message"], "La ubicación no es válida")

    def test_bool_coordinates_are_rejected(self):
        status, _body = response_for(
            json.dumps(context_body(location={"latitude": True, "longitude": 1})).encode(),
            DISABLED,
        )
        self.assertEqual(status, 400)

    def test_empty_list_says_there_is_nothing_nearby(self):
        status, body = response_for(json.dumps(context_body(pointsOfInterest=[])).encode(), DISABLED)
        self.assertEqual(status, 200)
        self.assertIn("nada especial", body["reply"])

    def test_custom_area_is_told_before_a_nearby_place(self):
        area = {
            "name": "Estadio Ciudad de Vicente López",
            "category": "custom-area",
            "description": "Es la cancha del Club Atlético Platense.",
            "latitude": -34.5405,
            "longitude": -58.4822,
        }
        raw = json.dumps(context_body(areas=[area])).encode()
        status, first = response_for(raw, DISABLED)
        self.assertEqual(status, 200)
        self.assertEqual(first["reply"], "Es la cancha del Club Atlético Platense.")
        self.assertEqual(first["interest"]["name"], "Estadio Ciudad de Vicente López")
        self.assertEqual(first["interest"]["category"], "custom-area")
        self.assertEqual([item["name"] for item in first["interests"]], ["Estadio Ciudad de Vicente López"])

        status, second = response_for(raw, DISABLED)
        self.assertEqual(status, 200)
        self.assertIn("Plaza de Mayo", second["reply"])
        self.assertNotIn("Platense", second["reply"])

    def test_a_danger_area_does_not_replace_the_place(self):
        raw = json.dumps(
            context_body(
                areas=[
                    {
                        "name": "Villa",
                        "category": "danger",
                        "description": "Estás entrando a una zona marcada.",
                        "latitude": -34.6,
                        "longitude": -58.38,
                    }
                ]
            )
        ).encode()
        status, body = response_for(raw, DISABLED)
        self.assertEqual(status, 200)
        self.assertIn("Plaza de Mayo", body["reply"])
        self.assertNotIn("zona marcada", body["reply"])

    def test_a_broken_custom_area_does_not_drop_the_place(self):
        parsed = parse_context(
            context_body(areas=["no", {"name": "Estadio", "category": "custom-area"}])
        )
        self.assertEqual(parsed.areas, [])
        status, body = response_for(
            json.dumps(context_body(areas=["no", {"category": "custom-area"}])).encode(),
            DISABLED,
        )
        self.assertEqual(status, 200)
        self.assertIn("Plaza de Mayo", body["reply"])

    def test_reply_mentions_the_place_and_direction(self):
        status, body = response_for(json.dumps(context_body()).encode(), DISABLED)
        self.assertEqual(status, 200)
        self.assertIn("Plaza de Mayo", body["reply"])
        self.assertIn("1810", body["reply"])
        self.assertNotIn("userId", body["reply"])
        interest = body["interest"]
        self.assertIsInstance(interest, dict)
        self.assertEqual(interest["name"], "Plaza de Mayo")
        self.assertEqual([item["name"] for item in body["interests"]], ["Plaza de Mayo"])

    def test_common_business_does_not_trigger_speech(self):
        places = [
            {
                "id": "farmacia",
                "name": "Farmacia del Centro",
                "latitude": -34.6038,
                "longitude": -58.3817,
                "category": "pharmacy",
                "distanceMeters": 15,
            }
        ]
        status, body = response_for(json.dumps(context_body(pointsOfInterest=places)).encode(), DISABLED)
        self.assertEqual(status, 200)
        self.assertEqual(body["reply"], "")
        self.assertEqual(body["interests"], [])

    def test_skips_a_business_and_speaks_about_the_monument(self):
        places = [
            {
                "id": "farmacia",
                "name": "Farmacia del Centro",
                "latitude": -34.6038,
                "longitude": -58.3817,
                "category": "pharmacy",
                "distanceMeters": 15,
            },
            context_body()["pointsOfInterest"][0],
        ]
        status, body = response_for(json.dumps(context_body(pointsOfInterest=places)).encode(), DISABLED)
        self.assertEqual(status, 200)
        self.assertIn("Plaza de Mayo", body["reply"])
        self.assertNotIn("Farmacia", body["reply"])
        self.assertEqual([item["name"] for item in body["interests"]], ["Plaza de Mayo"])

    def test_a_silent_guide_keeps_the_interest_pin_for_later(self):
        memory = MentionMemory()
        parsed = parse_context(context_body())
        silent = build_guide(parsed, DISABLED, user_id="visitante", memory=memory, speak=False)
        self.assertEqual(silent.reply, "")
        self.assertEqual([place.name for place in silent.places], ["Plaza de Mayo"])
        spoken = build_guide(parsed, DISABLED, user_id="visitante", memory=memory)
        self.assertIn("1810", spoken.reply)

    def test_plaza_zones_collapse_into_one_interest_and_one_story(self):
        places = [
            {
                "id": "plaza",
                "name": "Plaza San Martín",
                "latitude": -34.5950,
                "longitude": -58.3750,
                "category": "plaza",
                "distanceMeters": 40,
            },
            {
                "id": "juegos",
                "name": "Zona de juegos Plaza San Martín",
                "latitude": -34.5952,
                "longitude": -58.3751,
                "category": "park",
                "distanceMeters": 55,
            },
            {
                "id": "mascotas",
                "name": "Área de mascotas Plaza San Martín",
                "latitude": -34.5949,
                "longitude": -58.3748,
                "category": "park",
                "distanceMeters": 70,
            },
            {
                "id": "gym",
                "name": "Zona de entrenamiento Plaza San Martín",
                "latitude": -34.5951,
                "longitude": -58.3753,
                "category": "park",
                "distanceMeters": 65,
            },
        ]
        status, body = response_for(
            json.dumps(context_body(pointsOfInterest=places)).encode(),
            DISABLED,
        )
        self.assertEqual(status, 200)
        self.assertEqual([item["name"] for item in body["interests"]], ["Plaza San Martín"])
        self.assertEqual(body["interest"]["name"], "Plaza San Martín")
        reply = body["reply"]
        self.assertIn("Plaza San Martín", reply)
        self.assertTrue(any(word in reply for word in ("juegos", "mascotas", "entrenamiento")))
        self.assertNotIn("Zona de juegos Plaza San Martín", body["interests"][0]["name"])

    def test_a_repeated_place_stays_silent(self):
        memory = MentionMemory()
        parsed = parse_context(context_body())
        first = build_guide(parsed, DISABLED, user_id="visitante", memory=memory)
        second = build_guide(parsed, DISABLED, user_id="visitante", memory=memory)
        self.assertIn("1810", first.reply)
        self.assertEqual(second.reply, "")
        self.assertIn("1810", second.remembered)

    def test_known_place_includes_retrieved_history(self):
        status, body = response_for(json.dumps(context_body()).encode(), DISABLED)
        self.assertEqual(status, 200)
        self.assertIn("1810", body["reply"])
        self.assertIn("Revolución de Mayo", body["reply"])

    def test_unknown_place_does_not_borrow_nearby_history(self):
        places = [
            {
                "id": "kiosco",
                "name": "Kiosco El Rápido",
                "latitude": -34.6083,
                "longitude": -58.3722,
                "category": "store",
                "distanceMeters": 30,
            }
        ]
        status, body = response_for(json.dumps(context_body(pointsOfInterest=places)).encode(), DISABLED)
        self.assertEqual(status, 200)
        self.assertEqual(body["reply"], "")
        self.assertNotIn("1810", body["reply"])

    def test_same_name_far_away_does_not_match_the_corpus(self):
        places = [
            {
                "id": "otra-plaza",
                "name": "Plaza de Mayo",
                "latitude": 19.4326,
                "longitude": -99.1332,
                "category": "plaza",
                "distanceMeters": 80,
            }
        ]
        parsed = parse_context(context_body(pointsOfInterest=places))
        self.assertEqual(search_history(analyze_situation(parsed)), [])

    def test_broken_items_do_not_drop_a_valid_place(self):
        places = [{"name": ""}, context_body()["pointsOfInterest"][0]]
        parsed = parse_context(context_body(pointsOfInterest=places))
        self.assertEqual([place.name for place in parsed.places], ["Plaza de Mayo"])

    def test_unreadable_places_still_get_a_reply(self):
        status, body = response_for(
            json.dumps(context_body(pointsOfInterest=[{"name": 1}])).encode(),
            DISABLED,
        )
        self.assertEqual(status, 200)
        self.assertIn("no pude leer", body["reply"])


class ClusterTests(unittest.TestCase):
    def test_nearby_plaza_amenities_become_one_cluster(self):
        plaza = Place("Plaza San Martín", "plaza", 40, -34.5950, -58.3750)
        games = Place("Zona de juegos Plaza San Martín", "park", 55, -34.5952, -58.3751)
        dogs = Place("Área de mascotas Plaza San Martín", "park", 70, -34.5949, -58.3748)
        clusters = cluster_interests(
            [
                rank_place(games, "", False),
                rank_place(plaza, "Historia de la plaza.", False),
                rank_place(dogs, "", False),
            ]
        )
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0].primary.name, "Plaza San Martín")
        self.assertEqual(
            set(clusters[0].zone_labels),
            {"juegos", "mascotas"},
        )

    def test_far_places_stay_separate(self):
        plaza = Place("Plaza San Martín", "plaza", 40, -34.5950, -58.3750)
        other = Place("Plaza Italia", "plaza", 80, -34.5800, -58.4200)
        clusters = cluster_interests([rank_place(plaza, "", False), rank_place(other, "", False)])
        self.assertEqual({cluster.primary.name for cluster in clusters}, {"Plaza San Martín", "Plaza Italia"})

    def test_zone_label_reads_the_amenity(self):
        self.assertEqual(zone_label("Zona de entrenamiento Plaza San Martín"), "entrenamiento")
        self.assertEqual(zone_label("Área de mascotas"), "mascotas")


class GraphTests(unittest.TestCase):
    def test_jev_payload_is_a_noul_question_over_a_clean_state(self):
        request = build_request(
            {
                "poi_name": "Plaza de Mayo",
                "poi_category": "plaza",
                "in_curated_base": True,
                "recent_history": ["Obelisco"],
            }
        )
        self.assertEqual(
            set(request["state"]),
            {"poi_name", "poi_category", "in_curated_base", "recent_history"},
        )
        self.assertIs(request["state"]["in_curated_base"], True)
        question = request["questions"]["should_speak"]
        self.assertEqual(question["type"], "noul")
        self.assertIn("criteria", question)
        self.assertIn("true", question["criteria"])
        self.assertIn("false", question["criteria"])
        self.assertIn("no una empresa", CRITERIA_TRUE)
        self.assertIn("vista de pájaros", CRITERIA_FALSE)

    def test_scenic_business_name_is_not_a_place_to_narrate(self):
        agency = simulate_noul(
            {
                "poi_name": "Vista de Pájaros",
                "poi_category": "tourist_attraction",
                "in_curated_base": False,
                "recent_history": [],
            }
        )
        office = simulate_noul(
            {
                "poi_name": "Agencia de viajes Salvajes",
                "poi_category": "tourist_attraction",
                "in_curated_base": False,
                "recent_history": [],
            }
        )
        lookout = simulate_noul(
            {
                "poi_name": "Mirador de los Pájaros",
                "poi_category": "tourist_attraction",
                "in_curated_base": False,
                "recent_history": [],
            }
        )
        self.assertLess(agency, DISABLED.jev_threshold)
        self.assertLess(office, 0.1)
        self.assertGreater(lookout, DISABLED.jev_threshold)

    def test_pharmacy_ends_before_the_guide_node(self):
        graph = compile_graph(DISABLED)
        state = {
            "user_location": {"latitude": -34.6, "longitude": -58.38},
            "poi_data": {
                "name": "Farmacia del Centro",
                "category": "pharmacy",
                "description": "No usar.",
                "source": "",
                "latitude": -34.6,
                "longitude": -58.38,
            },
            "recent_history": [],
            "jev_confidence": 0.0,
            "should_trigger": False,
            "final_audio_script": "",
        }
        nodes: list[str] = []
        for update in graph.stream(state, stream_mode="updates"):
            nodes.extend(update)
        self.assertEqual(nodes, ["evaluator"])
        result = graph.invoke(state)
        self.assertFalse(result["should_trigger"])
        self.assertLessEqual(result["jev_confidence"], DISABLED.jev_threshold)
        self.assertEqual(result["final_audio_script"], "")
        self.assertNotIn("No usar", result["final_audio_script"])

    def test_monument_reaches_the_guide_node(self):
        graph = compile_graph(DISABLED)
        state = {
            "user_location": {"latitude": -34.6037, "longitude": -58.3816},
            "poi_data": {
                "name": "Plaza de Mayo",
                "category": "plaza",
                "description": "El 25 de mayo de 1810 empezó la Revolución de Mayo.",
                "source": "local",
                "latitude": -34.6083,
                "longitude": -58.3722,
            },
            "recent_history": [],
            "jev_confidence": 0.0,
            "should_trigger": False,
            "final_audio_script": "",
        }
        nodes: list[str] = []
        for update in graph.stream(state, stream_mode="updates"):
            nodes.extend(update)
        self.assertEqual(nodes, ["evaluator", "guide"])
        result = graph.invoke(state)
        self.assertTrue(result["should_trigger"])
        self.assertGreater(result["jev_confidence"], DISABLED.jev_threshold)
        self.assertIn("1810", result["final_audio_script"])


MUSEO_MITRE = {
    "id": "museo-mitre",
    "name": "Museo Mitre",
    "latitude": -34.6052,
    "longitude": -58.3745,
    "category": "museum",
    "distanceMeters": 90,
}


def wiki_payload(title, extract, lat, lon):
    return {
        "query": {
            "pages": [
                {"title": title, "extract": extract, "coordinates": [{"lat": lat, "lon": lon}]},
            ]
        }
    }


class WebContextTests(unittest.TestCase):
    def setUp(self) -> None:
        SHARED_MEMORY.clear()

    def test_listed_place_never_goes_to_the_web(self):
        with patch("agent.graph.lookup_web") as web:
            status, body = response_for(json.dumps(context_body()).encode(), DISABLED)
        self.assertEqual(status, 200)
        self.assertIn("1810", body["reply"])
        web.assert_not_called()

    def test_unlisted_monument_uses_the_web_fact(self):
        found = WebPassage("Museo Mitre", "Fue la casa de Bartolomé Mitre y abrió como museo en 1907.")
        with patch("agent.graph.lookup_web", return_value=found) as web:
            status, body = response_for(
                json.dumps(context_body(pointsOfInterest=[MUSEO_MITRE])).encode(),
                DISABLED,
            )
        self.assertEqual(status, 200)
        self.assertIn("1907", body["reply"])
        web.assert_called_once()

    def test_unlisted_monument_without_web_result_still_speaks(self):
        with patch("agent.graph.lookup_web", return_value=None):
            status, body = response_for(
                json.dumps(context_body(pointsOfInterest=[MUSEO_MITRE])).encode(),
                DISABLED,
            )
        self.assertEqual(status, 200)
        self.assertIn("Museo Mitre", body["reply"])

    def test_pharmacy_is_never_looked_up(self):
        places = [
            {
                "id": "farmacia",
                "name": "Farmacia del Centro",
                "latitude": -34.6038,
                "longitude": -58.3817,
                "category": "pharmacy",
                "distanceMeters": 15,
            }
        ]
        with patch("agent.graph.lookup_web") as web:
            response_for(json.dumps(context_body(pointsOfInterest=places)).encode(), DISABLED)
        web.assert_not_called()

    def test_page_with_the_same_name_and_place_is_picked(self):
        payload = wiki_payload(
            "Museo Mitre",
            "El Museo Mitre (en la casa de Mitre) es un museo histórico de Buenos Aires. Abrió en 1907.",
            -34.6053,
            -58.3746,
        )
        found = pick_page(payload, "Museo Mitre", -34.6052, -58.3745)
        self.assertIsNotNone(found)
        self.assertNotIn("(", found.text)
        self.assertIn("1907", found.text)

    def test_a_page_too_far_away_is_ignored(self):
        payload = wiki_payload("Museo Mitre", "Un museo con el mismo nombre en otro lado.", -34.62, -58.40)
        self.assertIsNone(pick_page(payload, "Museo Mitre", -34.6052, -58.3745))

    def test_a_nearby_page_with_another_name_is_ignored(self):
        payload = wiki_payload("Banco de la Nación", "La sede central del banco está en la city porteña.", -34.6053, -58.3746)
        self.assertIsNone(pick_page(payload, "Museo Mitre", -34.6052, -58.3745))

    def test_one_shared_word_is_not_a_match(self):
        self.assertEqual(match_score("Plaza San Martín", "Plaza de Mayo", 40), 0)

    def test_no_time_left_skips_the_web(self):
        with patch("agent.graph.lookup_web") as web:
            reply = build_reply(
                parse_context(context_body(pointsOfInterest=[MUSEO_MITRE])),
                DISABLED,
                memory=MentionMemory(),
                deadline=Deadline(0.0),
            )
        self.assertEqual(reply, "")
        web.assert_not_called()


class CityKnowledgeTests(unittest.TestCase):
    def test_both_cities_load_with_caba_places(self):
        counts = load_index().counts()
        self.assertEqual(set(counts), {"caba", "vicente-lopez"})
        self.assertEqual(counts["caba"], 16)

    def test_a_point_on_general_paz_looks_at_both_cities(self):
        self.assertEqual(load_index().cities_near(-34.540, -58.480), ["caba", "vicente-lopez"])

    def test_olivos_only_looks_at_vicente_lopez(self):
        self.assertEqual(load_index().cities_near(-34.507, -58.490), ["vicente-lopez"])

    def test_a_city_without_data_finds_nothing_locally(self):
        index = load_index()
        self.assertEqual(index.cities_near(-31.4167, -64.1833), [])
        place = Place("Cabildo", "historical_landmark", 50, -31.4167, -64.1833)
        self.assertIsNone(index.search(place))

    def test_curated_vicente_lopez_place_is_spoken(self):
        SHARED_MEMORY.clear()
        place = {
            "id": "torre",
            "name": "Torre Ader",
            "latitude": -34.5340,
            "longitude": -58.5450,
            "category": "point_of_interest",
            "distanceMeters": 80,
        }
        body = context_body(location={"latitude": -34.5335, "longitude": -58.5445}, pointsOfInterest=[place])
        with patch("agent.graph.lookup_web") as web:
            status, response = response_for(json.dumps(body).encode(), DISABLED)
        self.assertEqual(status, 200)
        self.assertIn("217 escalones", response["reply"])
        web.assert_not_called()

    def test_a_vicente_lopez_place_is_found_from_its_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            shutil.copy(KNOWLEDGE_DIR / "cities.json", base / "cities.json")
            folder = base / "places" / "vicente-lopez"
            folder.mkdir(parents=True)
            (folder / "quinta-trabucco.json").write_text(
                json.dumps(
                    {
                        "id": "quinta-trabucco",
                        "title": "Quinta Trabucco",
                        "aliases": ["Quinta Trabucco"],
                        "latitude": -34.5206,
                        "longitude": -58.4886,
                        "text": "Texto de prueba.",
                    }
                ),
                encoding="utf-8",
            )
            index = read_knowledge(base)
        found = index.search(Place("Quinta Trabucco", "park", 60, -34.5208, -58.4884))
        self.assertIsNotNone(found)
        self.assertEqual(found.id, "vicente-lopez/quinta-trabucco")
        self.assertEqual(found.city, "vicente-lopez")

    def test_a_broken_place_file_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            shutil.copy(KNOWLEDGE_DIR / "cities.json", base / "cities.json")
            folder = base / "places" / "caba"
            folder.mkdir(parents=True)
            (folder / "roto.json").write_text("{no es json", encoding="utf-8")
            index = read_knowledge(base)
        self.assertEqual(index.counts()["caba"], 0)


class VoiceNameTests(unittest.TestCase):
    def test_mentions_place_with_accents(self):
        self.assertTrue(mentions_place("Vale la pena mirar la Plaza San Martín un rato.", "Plaza San Martín"))
        self.assertTrue(mentions_place("La plaza san martin tiene sombra.", "Plaza San Martín"))

    def test_wikipedia_style_name_is_not_enough(self):
        self.assertFalse(
            mentions_place(
                "La Plaza General San Martín es el principal espacio verde de Retiro.",
                "Plaza San Martín",
            )
        )

    def test_ensure_place_name_prefixes_when_missing(self):
        text = ensure_place_name(
            "La Plaza General San Martín es el principal espacio verde de Retiro.",
            "Plaza San Martín",
        )
        self.assertTrue(text.startswith("Plaza San Martín."))
        self.assertIn("Plaza San Martín", text)

    def test_ensure_place_name_keeps_text_when_present(self):
        original = "Si mirás para allá, está Plaza San Martín. Vale la pena parar."
        self.assertEqual(ensure_place_name(original, "Plaza San Martín"), original)


class SettingsTests(unittest.TestCase):
    def test_blank_key_disables_the_model(self):
        settings = settings_from({"MODEL_API_KEY": "  ", "MODEL_BASE_URL": "", "MODEL_NAME": ""})
        self.assertFalse(settings.model_enabled)

    def test_key_with_newline_is_rejected(self):
        settings = settings_from({"MODEL_API_KEY": "abc\nX-Injected: 1"})
        self.assertFalse(settings.model_enabled)

    def test_https_provider_is_accepted(self):
        settings = settings_from({"MODEL_API_KEY": "test-key", "MODEL_BASE_URL": "https://api.openai.com/v1"})
        self.assertTrue(settings.model_enabled)
        self.assertEqual(settings.model_base_url, "https://api.openai.com/v1")

    def test_remote_http_and_file_urls_are_rejected(self):
        for url in ("http://example.com/v1", "file:///etc/passwd", "https://user:secret@api.openai.com/v1"):
            settings = settings_from({"MODEL_API_KEY": "test-key", "MODEL_BASE_URL": url, "MODEL_NAME": "gpt-4o-mini"})
            self.assertFalse(settings.model_enabled, url)

    def test_loopback_http_is_accepted(self):
        settings = settings_from(
            {"MODEL_API_KEY": "test-key", "MODEL_BASE_URL": "http://127.0.0.1:11434/v1", "MODEL_NAME": "llama"}
        )
        self.assertTrue(settings.model_enabled)

    def test_disabled_model_still_returns_a_reply_without_network(self):
        parsed = parse_context(context_body())
        reply = build_reply(parsed, DISABLED)
        self.assertIn("Plaza de Mayo", reply)


class BodyLimitTests(unittest.TestCase):
    def test_missing_length_and_chunked_bodies_are_rejected(self):
        self.assertIsNotNone(body_length_error(None, None, length_count=0))
        self.assertIsNotNone(body_length_error("10", "chunked"))
        self.assertIsNotNone(body_length_error("10", None, length_count=2))

    def test_oversized_body_is_rejected(self):
        self.assertIn("demasiado grande", body_length_error("70000", None) or "")

    def test_normal_length_is_accepted(self):
        self.assertIsNone(body_length_error("128", None))


if __name__ == "__main__":
    unittest.main()
