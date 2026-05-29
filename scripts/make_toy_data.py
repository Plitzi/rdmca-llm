#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import sys, os
_venv = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".venv", "bin", "python")
if os.path.exists(_venv) and os.path.abspath(sys.executable) != os.path.abspath(_venv):
    os.execv(_venv, [_venv] + sys.argv)

"""
Genera un corpus sintético mínimo para probar el pipeline completo
sin descargar nada de internet.

Produce ~3MB por idioma en data/stage1_language/:
  toy_en.jsonl
  toy_es.jsonl

Uso:
  python scripts/make_toy_data.py
"""
import json
import random
from pathlib import Path

random.seed(42)

# ── Bloques de texto EN ────────────────────────────────────────────────────

EN_STORIES = [
    "The sun was rising over the mountains when Maria woke up. She stretched her arms and looked out the window. Birds were singing in the trees outside. It was going to be a beautiful day.",
    "John walked to the market every Saturday morning. He liked to buy fresh vegetables and bread. The market was always crowded with people and full of colors and smells.",
    "The library was quiet. Only the sound of turning pages could be heard. A young girl sat at the corner table, reading a book about stars and planets.",
    "It rained all afternoon. The streets were empty. The cats hid under the benches in the park. The puddles reflected the grey sky above.",
    "The old baker woke up at four in the morning to prepare the bread. He mixed the flour, water, salt, and yeast. Then he let the dough rest for an hour before putting it in the oven.",
    "The teacher wrote the question on the board: what is the capital of France? Most students raised their hands immediately. Paris, they said.",
    "The train arrived on time. The passengers gathered their bags and walked toward the exit. Outside, taxis were waiting in a long line.",
    "She learned to play the guitar when she was twelve years old. At first, her fingers hurt from pressing the strings. But after a few months she could play simple songs.",
    "The dog ran across the field, chasing a ball. His tail wagged from side to side. When he returned with the ball, he dropped it at his owner's feet and waited.",
    "Every evening, the family sat around the table for dinner. They talked about their day. The children described what they had learned at school.",
]

EN_SCIENCE = [
    "Water is made of two hydrogen atoms and one oxygen atom. Its chemical formula is H2O. Water freezes at zero degrees Celsius and boils at one hundred degrees Celsius.",
    "The Earth revolves around the Sun once every 365 days. This is what gives us our calendar year. The Earth also rotates on its own axis every 24 hours, giving us day and night.",
    "Plants make their own food through a process called photosynthesis. They absorb sunlight through their leaves. They also take in carbon dioxide from the air and water from the soil.",
    "Gravity is the force that pulls objects toward each other. The more mass an object has, the stronger its gravitational pull. The Earth's gravity keeps us on the ground.",
    "Sound travels through the air as waves of pressure. When something vibrates, it pushes the air around it, creating sound waves. These waves reach our ears and we hear the sound.",
    "The human body has 206 bones. Bones protect our organs and allow us to move. They are connected at joints, which are held together by ligaments.",
    "Electricity flows through conductors like copper wire. Insulators like rubber and plastic prevent electricity from passing through them. This is why electrical wires are covered in plastic.",
    "The speed of light is approximately 300,000 kilometers per second. Light from the Sun takes about 8 minutes to reach the Earth. Light from the nearest star takes about 4 years.",
    "DNA carries the genetic information of living things. It is found in the nucleus of cells. DNA is shaped like a twisted ladder, which scientists call a double helix.",
    "The water cycle describes how water moves through the environment. Water evaporates from oceans and lakes. It forms clouds and falls as rain or snow, then flows back to the ocean.",
]

EN_REASONING = [
    "If all dogs are animals, and Max is a dog, then Max is an animal. This is a simple example of logical deduction.",
    "Maria has 12 apples. She gives 4 to her friend and eats 2 herself. How many apples does she have left? She has 6 apples left.",
    "A store sells shirts for 20 dollars each. A customer buys 3 shirts. How much does the customer pay? The customer pays 60 dollars.",
    "If today is Monday, what day will it be in three days? In three days it will be Thursday.",
    "The temperature in the morning was 15 degrees. By afternoon it had risen by 8 degrees. What was the afternoon temperature? It was 23 degrees.",
    "A car travels at 60 kilometers per hour. How far does it travel in 2 hours? It travels 120 kilometers.",
    "There are 30 students in a class. 18 are girls. How many are boys? There are 12 boys.",
    "A recipe needs 3 eggs to make 12 cookies. How many eggs are needed to make 24 cookies? You need 6 eggs.",
    "If a book has 200 pages and you read 25 pages per day, how many days will it take to finish? It will take 8 days.",
    "A triangle has three sides. A square has four sides. An octagon has eight sides. How many sides do a triangle and a square have together? They have seven sides together.",
]

# ── Bloques de texto ES ────────────────────────────────────────────────────

ES_STORIES = [
    "El sol salía sobre las montañas cuando María se despertó. Estiró los brazos y miró por la ventana. Los pájaros cantaban en los árboles afuera. Iba a ser un día hermoso.",
    "Juan caminaba al mercado cada sábado por la mañana. Le gustaba comprar verduras frescas y pan. El mercado siempre estaba lleno de gente y de colores y aromas.",
    "La biblioteca estaba tranquila. Solo se escuchaba el ruido de las páginas al pasar. Una niña estaba sentada en la mesa del rincón, leyendo un libro sobre estrellas y planetas.",
    "Llovió toda la tarde. Las calles estaban vacías. Los gatos se escondieron debajo de los bancos del parque. Los charcos reflejaban el cielo gris.",
    "El viejo panadero se levantaba a las cuatro de la mañana para preparar el pan. Mezclaba la harina, el agua, la sal y la levadura. Luego dejaba reposar la masa una hora antes de meterla al horno.",
    "La maestra escribió la pregunta en el pizarrón: ¿cuál es la capital de Francia? La mayoría de los estudiantes levantaron la mano de inmediato. París, dijeron.",
    "El tren llegó a tiempo. Los pasajeros recogieron sus bolsos y caminaron hacia la salida. Afuera, los taxis esperaban en una larga fila.",
    "Ella aprendió a tocar la guitarra cuando tenía doce años. Al principio, los dedos le dolían de presionar las cuerdas. Pero después de unos meses podía tocar canciones sencillas.",
    "El perro corrió por el campo persiguiendo una pelota. La cola le movía de un lado al otro. Cuando regresó con la pelota, la dejó a los pies de su dueño y esperó.",
    "Cada noche, la familia se sentaba alrededor de la mesa a cenar. Hablaban de su día. Los niños contaban lo que habían aprendido en la escuela.",
]

ES_SCIENCE = [
    "El agua está formada por dos átomos de hidrógeno y uno de oxígeno. Su fórmula química es H2O. El agua se congela a cero grados Celsius y hierve a cien grados Celsius.",
    "La Tierra gira alrededor del Sol una vez cada 365 días. Esto es lo que nos da el año del calendario. La Tierra también rota sobre su propio eje cada 24 horas, dándonos el día y la noche.",
    "Las plantas producen su propio alimento mediante un proceso llamado fotosíntesis. Absorben la luz solar a través de sus hojas. También toman dióxido de carbono del aire y agua del suelo.",
    "La gravedad es la fuerza que atrae a los objetos entre sí. Cuanta más masa tiene un objeto, más fuerte es su atracción gravitacional. La gravedad de la Tierra nos mantiene en el suelo.",
    "El sonido viaja por el aire como ondas de presión. Cuando algo vibra, empuja el aire a su alrededor, creando ondas sonoras. Estas ondas llegan a nuestros oídos y escuchamos el sonido.",
    "El cuerpo humano tiene 206 huesos. Los huesos protegen nuestros órganos y nos permiten movernos. Están conectados en articulaciones, que son sostenidas por ligamentos.",
    "La electricidad fluye por conductores como el alambre de cobre. Los aislantes como el caucho y el plástico evitan que la electricidad los atraviese. Por eso los cables eléctricos están cubiertos de plástico.",
    "La velocidad de la luz es de aproximadamente 300.000 kilómetros por segundo. La luz del Sol tarda unos 8 minutos en llegar a la Tierra. La luz de la estrella más cercana tarda unos 4 años.",
    "El ADN transporta la información genética de los seres vivos. Se encuentra en el núcleo de las células. El ADN tiene forma de escalera retorcida, que los científicos llaman doble hélice.",
    "El ciclo del agua describe cómo el agua se mueve por el medio ambiente. El agua se evapora de los océanos y los lagos. Forma nubes y cae como lluvia o nieve, luego vuelve al océano.",
]

ES_REASONING = [
    "Si todos los perros son animales y Max es un perro, entonces Max es un animal. Este es un ejemplo sencillo de deducción lógica.",
    "María tiene 12 manzanas. Le da 4 a su amiga y se come 2 ella misma. ¿Cuántas manzanas le quedan? Le quedan 6 manzanas.",
    "Una tienda vende camisetas a 20 pesos cada una. Un cliente compra 3 camisetas. ¿Cuánto paga el cliente? El cliente paga 60 pesos.",
    "Si hoy es lunes, ¿qué día será en tres días? En tres días será jueves.",
    "La temperatura de la mañana era de 15 grados. Para la tarde había subido 8 grados. ¿Cuál era la temperatura de la tarde? Era de 23 grados.",
    "Un auto viaja a 60 kilómetros por hora. ¿Cuánto recorre en 2 horas? Recorre 120 kilómetros.",
    "Hay 30 estudiantes en una clase. 18 son niñas. ¿Cuántos son niños? Hay 12 niños.",
    "Una receta necesita 3 huevos para hacer 12 galletas. ¿Cuántos huevos se necesitan para hacer 24 galletas? Se necesitan 6 huevos.",
    "Si un libro tiene 200 páginas y leés 25 páginas por día, ¿cuántos días tardás en terminarlo? Tardás 8 días.",
    "Un triángulo tiene tres lados. Un cuadrado tiene cuatro lados. Un octágono tiene ocho lados. ¿Cuántos lados tienen juntos un triángulo y un cuadrado? Tienen siete lados juntos.",
]


def expand(blocks: list[str], target_docs: int) -> list[str]:
    """Repite y varía los bloques hasta llegar a target_docs documentos."""
    docs = []
    while len(docs) < target_docs:
        for text in blocks:
            docs.append(text)
            if len(docs) >= target_docs:
                break
        # Segunda pasada con variaciones menores
        for text in blocks:
            words = text.split()
            if len(words) > 6:
                # pequeña variación: mover primera oración al final
                sentences = text.split(". ")
                if len(sentences) > 1:
                    rotated = ". ".join(sentences[1:] + [sentences[0]])
                    docs.append(rotated)
            if len(docs) >= target_docs:
                break
    return docs[:target_docs]


def write(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    size_kb = path.stat().st_size // 1024
    print(f"  {path}  ({len(records):,} docs · {size_kb} KB)")


def main() -> None:
    print("Generating toy corpus (no download needed)…")

    N = 2000   # documentos por idioma — da ~3 MB por idioma

    en_all = expand(EN_STORIES + EN_SCIENCE + EN_REASONING, N)
    es_all = expand(ES_STORIES + ES_SCIENCE + ES_REASONING, N)

    en_records = [{"text": t, "lang": "en"} for t in en_all]
    es_records = [{"text": t, "lang": "es"} for t in es_all]

    write(en_records, Path("data/stage1_language/toy_en.jsonl"))
    write(es_records, Path("data/stage1_language/toy_es.jsonl"))

    total_chars = sum(len(r["text"]) for r in en_records + es_records)
    print(f"\n  Total: {len(en_records)+len(es_records):,} docs · "
          f"~{total_chars//4500:.0f}K tokens estimados")
    print("\nNext steps:")
    print("  python scripts/train_tokenizer.py --sample_mb 5")
    print("  python train_stage.py --stage 1 --config configs/rdmca_t2_toy.yaml")


if __name__ == "__main__":
    main()
