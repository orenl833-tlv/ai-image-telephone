ניסוי קטן שבניתי לאחרונה: עד כמה AI באמת יודע לתאר תמונה במילים?

ה-pipeline:
🖼️ מודל Vision מקבל ציור ומתאר אותו ב-40 מילים בלבד
🎨 מודל Image Generation מצייר אותו מחדש רק לפי הטקסט
⚖️ מודל שלישי (LLM-as-a-Judge) משווה למקור ונותן ציון ב-8 מדדים: קומפוזיציה, צבע, תאורה, תנוחות ועוד

השוויתי 6 מודלים (GPT, Gemini, Grok, Qwen) דרך OpenRouter API.

בהתחלה בדקתי על ציורים מפורסמים, ו"נערה עם עגיל פנינה" קיבלה 8.2/10. מסתבר שמודל התמונה פשוט זיהה את הציור וצייר אותו מהזיכרון 😅
אז עברתי לציורים פחות מוכרים, כדי שהמודלים לא יוכלו "לרמות", והוספתי בדיקת data leakage אוטומטית.

מה שהכי מעניין אותי: זו בעצם שיטת דחיסה. תמונה של מאות KB הופכת ל-40 מילים, ומהן משחזרים אותה. Lossy compression בגרסת AI.

כמה תובנות:
• מודלים זולים כמעט לא נופלים מהיקרים
• כשהמודלים קיבלו את מדדי ההערכה מראש, Grok קפץ למקום הראשון
• כ-$0.08 למודל לציור, עם cost tracking על כל קריאת API

Stack: Python, REST APIs, parallel processing, Prompt Engineering, Claude Code

אני בתחילת הדרך כ-Fullstack Developer ואשמח לשמוע על משרות entry level 🙌

#AI #LLM #GenerativeAI #Python #FullStack #PromptEngineering #OpenToWork #JuniorDeveloper


<!-- Attach image: linkedin_grades.png -->
