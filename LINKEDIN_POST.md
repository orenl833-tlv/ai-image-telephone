ניסוי קטן שבניתי לאחרונה: עד כמה AI באמת יודע לתאר תמונה במילים?

ה-pipeline:
🖼️ מודל Vision מקבל ציור ומתאר אותו ב40 מילים מקס
🎨 מודל Image Generation מצייר אותו מחדש רק לפי הטקסט
⚖️ מודל שלישי (LLM-as-a-Judge) משווה למקור ונותן ציון ב-8 מדדים: צבע שולט, הבעות פנים, קומפוזיציה ועוד

השוויתי 6 מודלים (GPT, Gemini, Grok, Qwen) דרך OpenRouter API.

ההשוואות היו על ציורים פחות מוכרים, כי על ציור מוכר ייתכן שהמודל יזהה מה הציור המקורי וירמה.
והוספתי בדיקת data leakage אוטומטית כדי לוודא את העניין.

מה שהכי מעניין אותי: זו בעצם שיטת דחיסה. תמונה של מאות KB הופכת ל-40 מילים, ומהן משחזרים אותה. Lossy compression בגרסת AI.

כמה תובנות:
• מודלים זולים כמעט לא נופלים מהיקרים
• כשהמודלים קיבלו את מדדי ההערכה מראש, Grok קפץ למקום הראשון
• כ-$0.08 למודל לציור, עם cost tracking על כל קריאת API

בשבוע הבא אנסה לעשות את אותו הדבר עם סרטונים קצרים ~

Stack: Python, REST APIs, parallel processing, Prompt Engineering, Claude Code


#AI #LLM #GenerativeAI #Python #FullStack #PromptEngineering #JuniorDeveloper

<!-- Attach images: linkedin_metrics.png (per-metric grades), linkedin_grades.png (with vs without the grading criteria) -->
