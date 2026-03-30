import re

file_path = '/home/pc/Desktop/pfe/presentation_pfe.html'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Remove slide-header
content = re.sub(r'<div class="slide-header">.*?</div>\n?', '', content)

# 2. Extract slides
# We split by </section> to get individual slides
slides = re.split(r'(</section>)', content)

# Group them into (content, separator) pairs
slide_blocks = []
for i in range(0, len(slides)-1, 2):
    slide_blocks.append(slides[i] + slides[i+1])
# Trailing content
tail = slides[-1] if len(slides) % 2 != 0 else ""

def get_footer_num(block):
    match = re.search(r'Diapositive (\d+)', block)
    return int(match.group(1)) if match else None

# Map current footers to blocks
slide_dict = {}
for i, block in enumerate(slide_blocks):
    num = get_footer_num(block)
    if num is not None:
        slide_dict[num] = i

# We need to reorder and renumber
# Sequence of footer numbers in the NEW order:
# 1-13 stays same (maybe shifted in number? No, "depuis la page 14")
# User says "decaler tous +1 page depuis la page 14"
# This means:
# Old 1-13 -> No change in number? Or 13 becomes 14?
# "14 titre visualisation" -> This suggests Old 13 becomes New 14.
# So all 1 to N become 2 to N+1? No, then 1 becomes 2. 
# Usually "decaler +1 depuis X" means X becomes X+1.
# If Page 14 title is Visualisation, and Old 13 was Visualisation, then 13 -> 14.
# Let's assume ALL footers N become N+1 from some point or from the start.
# If I start from Footer 1:
# 1->2, 2->3, ..., 13->14 (Visualisation), 14->15 (Contrib), 15->16 (Contrib), 16->17 (Socle), 20->18 (Socle), 17->19 (LSTM), 18->20 (Trans), 19->21 (CNN)
# And the ones after 20 (21-30) also shift? 21->22, ..., 30->31.

new_slide_order = []
for n in range(1, 13):
    if n in slide_dict:
        new_slide_order.append((slide_dict[n], n + 1))

# Footer 13 becomes 14
if 13 in slide_dict:
    new_slide_order.append((slide_dict[13], 14))

# Footer 14 becomes 15
if 14 in slide_dict:
    new_slide_order.append((slide_dict[14], 15))

# Footer 15 becomes 16
if 15 in slide_dict:
    new_slide_order.append((slide_dict[15], 16))

# Footer 16 becomes 17
if 16 in slide_dict:
    new_slide_order.append((slide_dict[16], 17))

# Footer 20 moves to 18
if 20 in slide_dict:
    new_slide_order.append((slide_dict[20], 18))

# Footer 17 becomes 19
if 17 in slide_dict:
    new_slide_order.append((slide_dict[17], 19))

# Footer 18 becomes 20
if 18 in slide_dict:
    new_slide_order.append((slide_dict[18], 20))

# Footer 19 becomes 21
if 19 in slide_dict:
    new_slide_order.append((slide_dict[19], 21))

# Shift footers 21 to end
for n in range(21, 50): # Assuming max 50 slides
    if n in slide_dict:
        new_slide_order.append((slide_dict[n], n + 1))

# Handle title page (footer 1 or no footer)
# Footer 1 (Diapositive 1) might be the first content slide.
# The title page (TP) in this file actually doesn't have a slide-footer div.
# Let's check the first slide footer.
# Line 283: Diapositive 2.
# So footer starts at 2.
# If footer 2 becomes 3, etc.
# User said "decaler tous +1 page".

# Assemble the new block sequence
final_blocks = []
current_idx = 0
# Add blocks that were before the first numbered slide
while current_idx < len(slide_blocks) and get_footer_num(slide_blocks[current_idx]) is None:
    final_blocks.append(slide_blocks[current_idx])
    current_idx += 1

# Add reordered/renumbered slides
for old_idx, new_num in new_slide_order:
    block = slide_blocks[old_idx]
    # Update footer
    block = re.sub(r'Diapositive \d+', f'Diapositive {new_num}', block)
    
    # Update Titles as per user request
    if new_num == 14:
        block = re.sub(r'<h2>.*?</h2>', '<h2>Visualisation des résultats</h2>', block)
    elif new_num in [15, 16]:
        block = re.sub(r'<h2>.*?</h2>', '<h2>Notre contribution</h2>', block)
    elif new_num in [17, 18]:
        block = re.sub(r'<h2>.*?</h2>', '<h2>pipeline de prétraitement (socle commun)</h2>', block)
    elif new_num == 19:
        block = re.sub(r'<h2>.*?</h2>', '<h2>Prétraitements Spécifiques LSTM</h2>', block)
    elif new_num == 20:
        block = re.sub(r'<h2>.*?</h2>', '<h2>Prétraitements Spécifiques Transformer</h2>', block)
    elif new_num == 21:
        block = re.sub(r'<h2>.*?</h2>', '<h2>Prétraitements Spécifiques CNN-LSTM</h2>', block)
        
    final_blocks.append(block)

# Construct final content
final_content = "".join(final_blocks) + tail

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(final_content)

print("Slide reordering and header removal complete.")
