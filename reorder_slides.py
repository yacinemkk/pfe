import re

file_path = '/home/pc/Desktop/pfe/presentation_pfe.html'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Global shift of footers: Diapositive X -> Diapositive X+1
# We do this from largest to smallest to avoid double increment issues if we were doing simple replaces,
# but using a regex with a lambda is better.
def shift_footer(match):
    num = int(match.group(1))
    # Decaler tous +1 page
    return f'Diapositive {num + 1}'

new_content = re.sub(r'Diapositive (\d+)', shift_footer, content)

# 2. Extract slides to reorder
# We need to find the sections.
# Slide 16 (new 17): Modélisation Temporelle
# Slide 20 (new 21): Génération de Tenseurs -> Should be moved to Page 18.
# Slide 17 (new 18): LSTM -> Should be Page 19.
# Slide 18 (new 19): Transformer -> Should be Page 20.
# Slide 19 (new 20): CNN-LSTM -> Should be Page 21.

sections = re.split(r'(<section[^>]*>)', new_content)
# sections[0] is header
# sections[1] is <section>
# sections[2] is content...

# Let's count sections based on footers.
slide_map = {}
current_footer = None
for i in range(len(sections)):
    footer_match = re.search(r'Diapositive (\d+)', sections[i])
    if footer_match:
        slide_map[int(footer_match.group(1))] = i

# Proposed reorder:
# ... 16 (stays 16), 17 (Socle), 18 (NEW Tenseurs), 19 (LSTM), 20 (Transformer), 21 (CNN-LSTM)

# Slide currently at Footer 17 (Modélisation Temporelle)
# Slide currently at Footer 21 (Génération de Tenseurs) -> Move to after Footer 17.

idx_tenseurs = slide_map[21]
idx_socle = slide_map[17]

# Extract tenseurs section
# Note: sections is [header, <section1>, content1, <section2>, content2, ...]
# So tenseurs is sections[idx_tenseurs-1] + sections[idx_tenseurs]
tenseurs_full = sections[idx_tenseurs-1] + sections[idx_tenseurs]

# Remove it from the original list
del sections[idx_tenseurs]
del sections[idx_tenseurs-1]

# Re-locate socle (its index might have changed if idx_tenseurs was before it, but it's 21 > 17)
idx_socle = -1
for i in range(len(sections)):
    if 'Diapositive 17' in sections[i]:
        idx_socle = i
        break

# Insert after socle
sections.insert(idx_socle + 1, tenseurs_full)

# Now re-number the shifted items
# Previous Footer 18 (LSTM) becomes 19.
# Previous Footer 19 (Transformer) becomes 20.
# Previous Footer 20 (CNN-LSTM) becomes 21.
# Tenseurs (moved) should become 18.

content_final = "".join(sections)

def fix_renumber(match):
    num = int(match.group(1))
    if num == 18: return 'Diapositive 19'
    if num == 19: return 'Diapositive 20'
    if num == 20: return 'Diapositive 21'
    if num == 21: return 'Diapositive 18' # The moved one
    return f'Diapositive {num}'

# This is tricky because content_final has the new numbers already (18, 19, 20, 21).
# I should have done re-numbering inside the section strings.

# Let's try a different approach.
# Renumber footers for the specific slides.
content_final = content_final.replace('Diapositive 18', 'TEMP_D19')
content_final = content_final.replace('Diapositive 19', 'TEMP_D20')
content_final = content_final.replace('Diapositive 20', 'TEMP_D21')
content_final = content_final.replace('Diapositive 21', 'Diapositive 18')
content_final = content_final.replace('TEMP_D19', 'Diapositive 19')
content_final = content_final.replace('TEMP_D20', 'Diapositive 20')
content_final = content_final.replace('TEMP_D21', 'Diapositive 21')

# Fix Titles
content_final = content_final.replace('<h2>Visualisation des résultats</h2>', 'TEMP_TITLE_VIS')
content_final = content_final.replace('<h2>Notre contribution</h2>', 'TEMP_TITLE_CONTRIB')
content_final = content_final.replace('<h2>Modélisation Temporelle : le socle commun</h2>', 'TEMP_TITLE_SOCLE')
content_final = content_final.replace('<h2>Prétraitements Spécifiques LSTM</h2>', 'TEMP_TITLE_LSTM')
content_final = content_final.replace('<h2>Prétraitements Spécifiques Transformer</h2>', 'TEMP_TITLE_TRANS')
content_final = content_final.replace('<h2>Prétraitements Spécifiques CNN-LSTM</h2>', 'TEMP_TITLE_CNN')
content_final = content_final.replace('<h2>Génération de Tenseurs</h2>', 'TEMP_TITLE_TENSEURS')

# User titles:
# 14: Visualisation des résultats
# 15: Notre contribution
# 16: Notre contribution
# 17: pipeline de pretraitement (socle commun)
# 19: Prétraitements Spécifiques LSTM
# 20: Prétraitements Spécifiques Transformer
# 21: Prétraitements Spécifiques CNN-LSTM

# We need to map them back to the Footers.
# Footer 14 (Slide 13): Visualisation
# Footer 15 (Slide 14): Notre contribution
# Footer 16 (Slide 15): Notre contribution
# Footer 17 (Slide 16): pipeline de pretraitement (socle commun)
# Footer 18 (Moved Slide 20): pipeline de pretraitement (socle commun)
# Footer 19 (Slide 17): Prétraitements Spécifiques LSTM
# Footer 20 (Slide 18): Prétraitements Spécifiques Transformer
# Footer 21 (Slide 19): Prétraitements Spécifiques CNN-LSTM

# Split again to be safe
final_sections = re.split(r'(<section[^>]*>)', content_final)
for i in range(len(final_sections)):
    # Footer 14
    if 'Diapositive 14' in final_sections[i]:
        final_sections[i] = re.sub(r'TEMP_TITLE_[A-Z]+', '<h2>Visualisation des résultats</h2>', final_sections[i])
    # Footer 15
    elif 'Diapositive 15' in final_sections[i]:
         final_sections[i] = re.sub(r'TEMP_TITLE_[A-Z]+', '<h2>Notre contribution</h2>', final_sections[i])
    # Footer 16
    elif 'Diapositive 16' in final_sections[i]:
         final_sections[i] = re.sub(r'TEMP_TITLE_[A-Z]+', '<h2>Notre contribution</h2>', final_sections[i])
    # Footer 17
    elif 'Diapositive 17' in final_sections[i]:
         final_sections[i] = re.sub(r'TEMP_TITLE_[A-Z]+', '<h2>pipeline de prétraitement (socle commun)</h2>', final_sections[i])
    # Footer 18
    elif 'Diapositive 18' in final_sections[i]:
         final_sections[i] = re.sub(r'TEMP_TITLE_[A-Z]+', '<h2>pipeline de prétraitement (socle commun)</h2>', final_sections[i])
    # Footer 19
    elif 'Diapositive 19' in final_sections[i]:
         final_sections[i] = re.sub(r'TEMP_TITLE_[A-Z]+', '<h2>Prétraitements Spécifiques LSTM</h2>', final_sections[i])
    # Footer 20
    elif 'Diapositive 20' in final_sections[i]:
         final_sections[i] = re.sub(r'TEMP_TITLE_[A-Z]+', '<h2>Prétraitements Spécifiques Transformer</h2>', final_sections[i])
    # Footer 21
    elif 'Diapositive 21' in final_sections[i]:
         final_sections[i] = re.sub(r'TEMP_TITLE_[A-Z]+', '<h2>Prétraitements Spécifiques CNN-LSTM</h2>', final_sections[i])
    else:
        # Restore other TEMP_TITLEs to their original if any left
        final_sections[i] = final_sections[i].replace('TEMP_TITLE_VIS', '<h2>Visualisation des résultats</h2>')
        final_sections[i] = final_sections[i].replace('TEMP_TITLE_CONTRIB', '<h2>Notre contribution</h2>')
        final_sections[i] = final_sections[i].replace('TEMP_TITLE_SOCLE', '<h2>Modélisation Temporelle : le socle commun</h2>')
        final_sections[i] = final_sections[i].replace('TEMP_TITLE_LSTM', '<h2>Prétraitements Spécifiques LSTM</h2>')
        final_sections[i] = final_sections[i].replace('TEMP_TITLE_TRANS', '<h2>Prétraitements Spécifiques Transformer</h2>')
        final_sections[i] = final_sections[i].replace('TEMP_TITLE_CNN', '<h2>Prétraitements Spécifiques CNN-LSTM</h2>')
        final_sections[i] = final_sections[i].replace('TEMP_TITLE_TENSEURS', '<h2>Génération de Tenseurs</h2>')

with open(file_path, 'w', encoding='utf-8') as f:
    f.write("".join(final_sections))
