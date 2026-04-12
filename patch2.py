import json

def main():
    path = '/home/pc/Desktop/pfe/greedy.ipynb'
    with open(path, 'r') as f:
        nb = json.load(f)
    changed = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            new_source = []
            for line in cell['source']:
                if "                scheduler.step()" in line and "for _ in range(start_epoch - 1):" in "".join(cell['source']):
                    if "optimizer.step()" not in line: # Prevent double replacement
                         new_source.append(line.replace("                scheduler.step()", "                optimizer.step()\n                scheduler.step()"))
                         changed = True
                         continue
                new_source.append(line)
            cell['source'] = new_source
            
            # Check single string source
            if isinstance(cell['source'], list) and len(cell['source']) == 1:
                text = cell['source'][0]
                if 'for _ in range(start_epoch - 1):\n                scheduler.step()\n' in text:
                    cell['source'] = [text.replace('for _ in range(start_epoch - 1):\n                scheduler.step()\n', 'for _ in range(start_epoch - 1):\n                optimizer.step()\n                scheduler.step()\n')]
                    changed = True

    if changed:
        with open(path, 'w') as f:
            json.dump(nb, f, indent=1)
        # Restore nice formatting
        with open(path, 'r') as f:
             content = f.read()
        content = content.replace('\\r\\n', '\\n')
        with open(path, 'w') as f:
             f.write(content)
        print("Patched successfully!")
    else:
        print("No changes needed or string not found.")

if __name__ == '__main__':
    main()
