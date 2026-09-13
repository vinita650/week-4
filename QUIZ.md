# Week-4 Deployment & Architecture Quiz

Quiz format: 8 short questions, one at a time, waiting for answer, graded 1 line each out of 8.

Topics covered:
- Deploying the app this week
- requirements.txt and pinned versions
- Secrets management
- What ships vs what stays
- The update loop
- Why the index ships in the repo

---

## Question 1: Deployment
**What service hosts your live app and how did you connect it to your GitHub repo?**

---

## Question 2: Pinned Versions
**Why does the CLAUDE.md file insist on exact version pinning with `==` instead of `>=` or `~=`?**

---

## Question 3: Secrets
**Where are your OpenAI API credentials stored so they never get committed to GitHub?**

---

## Question 4: What Ships
**Name 3 files/folders that ARE committed to your git repo and shipped to production.**

---

## Question 5: What Stays Local
**Name 2 files/folders that are in .gitignore so personal documents never ship to GitHub.**

---

## Question 6: The Update Loop
**When you push code to master, what happens automatically on Streamlit Cloud?**

---

## Question 7: Why Index Ships
**The chroma database is now in-memory (ephemeral). What problem does this solve vs. the old persistent database?**

---

## Question 8: RAG Fundamentals
**In one sentence: what is the core difference between RAG and a regular LLM?**

---

Grade yourself: 8/8 means you own this architecture.
