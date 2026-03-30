import re

file_path = '/home/pc/Desktop/pfe/presentation_pfe.html'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Split slides
slides = re.split(r'(</section>)', content)
slide_blocks = []
for i in range(0, len(slides)-1, 2):
    slide_blocks.append(slides[i] + slides[i+1])
tail = slides[-1] if len(slides) % 2 != 0 else ""

def get_footer_num(block):
    match = re.search(r'Diapositive (\d+)', block)
    return int(match.group(1)) if match else None

# Desired mapping of Footers to Titles and Images
# (Slide numbers here refer to the Diapositive number in footer)
mapping = {
    14: ("Visualisation des résultats", "image12.png"),
    15: ("Notre contribution", "image13.png"),
    16: ("Notre contribution", "image8.png"),
    17: ("pipeline de prétraitement (socle commun)", "image10.png"),
    18: ("pipeline de prétraitement (socle commun)", "image9.png"),
    19: ("Prétraitements Spécifiques LSTM", "image14.png"),
    20: ("Prétraitements Spécifiques Transformer", "image15.png"),
    21: ("Prétraitements Spécifiques CNN-LSTM", "image17.png"),
}

new_slide_blocks = []
for block in slide_blocks:
    num = get_footer_num(block)
    if num in mapping:
        title, img = mapping[num]
        # Replace title
        block = re.sub(r'<h2>.*?</h2>', f'<h2>{title}</h2>', block)
        # Replace image src (assuming one image per slide or replacing all)
        # We target presentation_assets/media/image[0-9]+.(png|jpg)
        block = re.sub(r'presentation_assets/media/image[0-9]+\.(png|jpg)', f'presentation_assets/media/{img}', block)
    new_slide_blocks.append(block)

final_content = "".join(new_slide_blocks) + tail
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(final_content)

print("Final title and image fix complete.")
