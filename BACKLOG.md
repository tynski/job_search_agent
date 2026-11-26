# Backlog

## Now
- [ ] job boards functionality not working
- [ ] CLI does not have help, what are the flags? I want to use specific source
- [ ] garbage collection - delete records older than 30 days

## Next
- [ ] Refactor all code, check tests coverage, remove sensitive information from code
- [ ] add Airbitenow

## Later
- [ ] deploy solution to production, make it stand alone CLI
- [ ] mail or telegram digest

## Ideas (not decided)
- Lightweight Machine Learning alternative for local scoring (e.g. embeddings + cross-encoder / semantic similarity or scikit-learn classifier for fast, zero-cost ranking without running slow LLMs on CPU)
- handle mulitjob functionality - architecture
- simple UI

## Done
- [x] Add README.md, describe how to use agent
- [x] make match easier to identify and write in CLI - maybe id or short name?
- [x] Delete Hacker News collector from code, tests, and documentation
- [x] Switch to cloud-only LLM scoring (removed slow local CPU Ollama setup)
- [x] Add company_boards collector (Greenhouse, Lever, Ashby, SmartRecruiters)
