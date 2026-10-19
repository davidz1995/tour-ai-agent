# Evolución de la base de conocimiento

Este documento describe cómo puede crecer el contexto histórico que usa el guía, desde el `places.json` actual hasta una base con fragmentos y embeddings. Cada etapa se puede hacer sola y no rompe el contrato con Node (`POST /context` → `reply`).

## Cómo funciona hoy

1. Jev decide si vale la pena hablar del lugar (`agent/jev.py`).
2. Si decide que sí, el guía busca contexto en este orden (`agent/graph.py`, nodo `guide`):
   - La base local, separada por ciudad (`agent/retrieve.py`).
   - Si no está, Wikipedia en español (`agent/web_context.py`).
   - Si no hay nada, habla sin dato histórico y no inventa fechas.
3. Un registro solo se acepta si el nombre coincide **y** está a menos de 450 metros del lugar. Es lo que evita que una Plaza de Mayo de otra ciudad herede la historia de Buenos Aires.

### La base local por ciudad

```text
agent/knowledge/
├── cities.json                  # id, nombre y rectángulo de cada ciudad
└── places/
    ├── caba/                    # 16 lugares, un archivo por lugar
    │   └── plaza-de-mayo.json
    └── vicente-lopez/           # lista para cargar
```

- Cada lugar es un JSON con `id`, `title`, `aliases`, `latitude`, `longitude` y `text`. La ciudad sale de la carpeta, y el `id` interno queda como `ciudad/lugar`, así dos «Plaza San Martín» de ciudades distintas no chocan.
- Para cada lugar que llega de Node, se miran solo las ciudades cuyo rectángulo, agrandado en unos 0,01 grados (alrededor de 1 km), contiene sus coordenadas. Cerca de la General Paz se miran CABA y Vicente López a la vez. En una ciudad que no está en `cities.json` no hay búsqueda local y se va directo a Wikipedia.
- Los rectángulos de `cities.json` son aproximados. Conviene ajustarlos con los límites oficiales y, más adelante, cambiarlos por el polígono real.
- **Sumar una ciudad:** agregarla en `cities.json`, crear `places/<id>/` y cargar los lugares. **Sumar un lugar:** crear su archivo en la carpeta de la ciudad. En los dos casos hay que reiniciar el agente.
- Un archivo inválido se saltea y se avisa al arrancar (`lugar inválido: ciudad/archivo.json`), sin tirar abajo el resto.
4. Todo el pedido tiene 8,5 segundos (`agent/deadline.py`), porque el orquestador corta a los 10.

El punto de cambio es `HistoryIndex.search()` en `agent/retrieve.py`: recibe un lugar y devuelve un texto o nada. Las etapas siguientes cambian qué hay detrás de esa función, no el grafo.

## Reglas que se mantienen en todas las etapas

- **Primero el lugar, después el significado.** Los embeddings no distinguen dos lugares con el mismo nombre en ciudades distintas. El filtro por distancia va siempre antes de la búsqueda semántica.
- **Nada pesado dentro del pedido.** Leer documentos, cortarlos y calcular embeddings se hace en un proceso aparte. En el pedido solo se consulta.
- **La base curada gana.** Internet es respaldo, nunca reemplaza un texto revisado.
- **Si no hay dato, no se inventa.** El guía habla igual, sin fechas ni episodios.

## Etapa 1: un Markdown por lugar

**Objetivo:** que el contenido lo pueda mantener alguien que no programa.

- Cambiar los JSON de `agent/knowledge/places/<ciudad>/` por `<lugar>.md`, un archivo por lugar:

  ```markdown
  ---
  id: plaza-de-mayo
  title: Plaza de Mayo
  aliases: [Plaza de Mayo]
  city: buenos-aires
  latitude: -34.608417
  longitude: -58.37225
  source: revisado
  reviewed_at: 2026-10-13
  ---

  Es la plaza mayor desde 1580, cuando Juan de Garay fundó la ciudad...

  ## Curiosidades

  ...
  ```

- Un script (`scripts/build_index.py`) lee los Markdown, valida los campos obligatorios y las coordenadas, y genera los JSON de cada ciudad. El agente sigue leyendo JSON, así que no cambia nada en el pedido.
- Los cambios de contenido se revisan en un pull request, como el código.
- Migrar los 16 relatos actuales y revisar los datos: se cargaron como semilla para probar el flujo.

**Listo cuando:** agregar un lugar es crear un `.md` y correr el script, sin tocar Python.

## Etapa 2: varios fragmentos por lugar

**Objetivo:** que un lugar tenga más de un dato y el guía no repita siempre el mismo.

- Cortar cada Markdown por secciones (historia, arquitectura, curiosidades) en fragmentos de 200 a 400 palabras, con algo de solapamiento.
- Cada fragmento conserva el `id` del lugar, sus coordenadas y la sección de la que sale.
- El guía elige, entre los fragmentos de ese lugar, uno que el usuario no haya escuchado. Hoy `agent/history.py` recuerda lugares; acá pasaría a recordar fragmentos.
- Sigue sin embeddings: la elección es por sección o al azar entre los no escuchados.

**Listo cuando:** el mismo usuario que pasa dos veces por un lugar escucha un dato distinto.

## Etapa 3: Supabase con pgvector

**Objetivo:** actualizar el contenido sin redeployar el agente y buscar por significado cuando haya volumen.

- Tabla de fragmentos en Supabase (ya lo usa el orquestador):

  ```sql
  create extension if not exists vector;

  create table knowledge_chunks (
    id          bigserial primary key,
    place_id    text not null,
    title       text not null,
    city        text,
    section     text,
    content     text not null,
    location    geography(point) not null,
    embedding   vector(1536),
    source      text not null,          -- revisado | wikipedia | pendiente
    reviewed_at date,
    updated_at  timestamptz default now()
  );

  create index on knowledge_chunks using gist (location);
  create index on knowledge_chunks using hnsw (embedding vector_cosine_ops);
  ```

- Un proceso de carga, fuera del pedido, lee los Markdown, corta los fragmentos, calcula los embeddings y hace upsert en la tabla. Corre en CI cuando cambia `knowledge/` o a mano.
- En el pedido, una sola consulta:
  1. filtra por distancia (`ST_DWithin`, 450 metros) y por coincidencia de nombre;
  2. entre esos fragmentos, ordena por similitud con la consulta y descarta los ya escuchados.
- El agente solo lee con una clave de lectura. La escritura queda para el proceso de carga.
- Medir la latencia: la consulta tiene que entrar en el presupuesto junto con Jev y el modelo. Si calcular el embedding de la consulta en cada pedido es caro, empezar ordenando por sección y dejar la similitud para cuando haga falta.

**Listo cuando:** se corrige un texto en Supabase y el próximo pedido ya lo usa, sin reiniciar el agente.

## Etapa 4: de internet a la base curada

**Objetivo:** que los lugares que la gente visita de verdad terminen en la base revisada.

- Cuando el agente encuentra algo en Wikipedia, lo guarda como fragmento con `source = 'pendiente'`, junto con el título y el enlace del artículo.
- Una vista o un panel muestra los pendientes ordenados por cuántas veces se pidieron.
- Alguien los revisa: los aprueba (pasan a `revisado`), los corrige o los descarta.
- Los descartados se recuerdan para no volver a buscar ese artículo para ese lugar.

**Listo cuando:** los lugares más pedidos ya no dependen de internet.

## Etapa 5: más ciudades y más idiomas

- La separación por ciudad ya existe en archivos. En Supabase, la ciudad pasa a ser una columna y el primer filtro de la consulta, antes que la distancia, para que la tabla crezca sin volver lenta la consulta.
- Cambiar los rectángulos de `cities.json` por los polígonos reales de cada ciudad o partido.
- Si se suma otro idioma, guardar el idioma en cada fragmento. Wikipedia de respaldo en el mismo idioma.
- Medir la cobertura por ciudad: qué parte de los lugares en los que Jev decide hablar tiene un dato curado.

## Qué medir en todas las etapas

- Tiempo de cada paso del pedido: Jev, búsqueda y modelo, con el total por debajo de los 10 segundos.
- De dónde salió el dato: base curada, Wikipedia o ninguno.
- Lugares en los que Jev decidió hablar y no hubo dato. Esa lista es la que se carga primero.
- Errores reportados de datos históricos, por fragmento, para corregirlos en la fuente.
