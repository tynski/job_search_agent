# Backlog

## Now
- [ ] CLI does not have help, what are the flags? I want to use specific source
- [ ] garbage collection - delete records older than 30 days

## Next
- [ ] More sources (research public API alternatives; pracuj.pl and theprotocol.it blocked by Cloudflare Turnstile 403)

## Later
- [ ] simple UI
- [ ] mail or telegram digest

## Ideas (not decided)
- Lightweight Machine Learning alternative for local scoring (e.g. embeddings + cross-encoder / semantic similarity or scikit-learn classifier for fast, zero-cost ranking without running slow LLMs on CPU)
- handle mulitjob functionality - architecture

## Done
- [x] Add README.md, describe how to use agent
- [x] make match easier to identify and write in CLI - maybe id or short name?
- [x] Delete Hacker News collector from code, tests, and documentation
- [x] Switch to cloud-only LLM scoring (removed slow local CPU Ollama setup)
