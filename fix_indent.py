import os

file_path = "c:/Users/aszx7/Desktop/taiwan-bot/taiwan-bot/app.py"
with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

in_sniper_block = False
indent_needed = False

new_lines = []
for line in lines:
    if 'elif st.session_state.current_page == "📊 台股大盤掃描":' in line:
        in_sniper_block = False
        indent_needed = False
        
    if in_sniper_block and indent_needed:
        if line.strip() == "":
            new_lines.append(line)
        else:
            new_lines.append("    " + line)
        continue

    if 'if not target_ticker:' in line:
        in_sniper_block = True
    elif in_sniper_block and 'else:' in line:
        indent_needed = True # From next line, we must indent by 4 spaces
        new_lines.append(line)
        continue
        
    new_lines.append(line)

with open(file_path, "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print("Indentation fixed.")
