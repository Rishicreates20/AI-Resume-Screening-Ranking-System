from screener.extract import extract_contacts, guess_name
from screener.llm.prompt import compact_for_llm
from screener.llm.rules_fallback import RulesAnalyzer


def test_github_profile_link_beats_repo_links_and_reserved_paths():
    text = "Jo Doe\nGitHub\nProjects\nBot | github.com/other-org/shared-lib\nsee github.com/features"
    links = ["https://github.com/jodoe", "https://github.com/jodoe/rag-bot", "https://github.com/features"]
    c = extract_contacts(text, links)
    assert c.github_username == "jodoe" and c.github_url == "https://github.com/jodoe"


def test_bare_github_link_and_label_without_username_give_none():
    c = extract_contacts("Jo Doe\nGitHub | LinkedIn", ["https://github.com/"])
    assert c.github_username is None


def test_github_url_wrapped_across_lines_is_rejoined():
    c = extract_contacts("Jo Doe\ngithub.com/\njodoe-dev\nSkills\nPython", [])
    assert c.github_username == "jodoe-dev"


def test_github_username_from_text_when_pdf_has_no_link_annotation():
    assert extract_contacts("Jo Doe | github.com/jodoe | jo@x.io", []).github_username == "jodoe"


def test_email_prefers_visible_text_and_name_falls_back_to_filename():
    c = extract_contacts("hello jo@x.io", ["mailto:stale@old.io"])
    assert c.email == "jo@x.io"
    assert guess_name("12345\n@@@", "candidate_07.pdf") == "Candidate 07"


def test_compact_for_llm_keeps_projects_and_drops_education_when_too_long():
    filler = "Coursework: " + "algorithms, " * 80
    text = (f"Jo Doe\njo@x.io\nEDUCATION\n{filler}\nCERTIFICATIONS\n{filler}\n"
            "PROJECTS\nAgent Desk\n- Built a LangGraph agent with tool calling and RAG over 10k documents.")
    assert compact_for_llm(text, 10_000) == text                       # fits: untouched
    short = compact_for_llm(text, 400)
    assert "LangGraph agent" in short and "algorithms" not in short and len(short) <= 400


def test_rules_summary_skips_company_date_role_header_lines(cfg):
    text = ("Jo Doe\nEXPERIENCE\nNetoAI Solutions\nJan 2026 - Present\nGenerative AI Engineer Intern\nRemote\n"
            "- Built a production LLM voice agent with tool calling on FreeSWITCH and SIP.\n")
    project = RulesAnalyzer(cfg).analyze(text).projects[0]
    assert project.summary.startswith("Built a production LLM voice agent")


def test_link_role_and_wrapped_lines_do_not_become_fake_projects():
    """Fragments like 'Role: ...', a GitHub link, a lone '|' or a sentence cut mid-clause stay inside
    their entry. Otherwise they are scored (and penalised) as separate 'projects'."""
    from screener.extract import split_entries

    section = ("Agent Desk | Python, LangGraph\n- Built a multi-agent workflow with tool calling over support tickets.\n"
               "Role: GenAI Developer\nGithub : https://github.com/jo/agent-desk\n|\n"
               "Performed tweet cleaning using CountVectorizer, and\n- Deployed on Cloud Run.\n"
               "Invoice Service | FastAPI\n- Parsed invoices with Celery.")
    entries = split_entries(section)
    assert [e["title"] for e in entries] == ["Agent Desk", "Invoice Service"]
    assert "Role: GenAI Developer" in entries[0]["text"]


def test_singular_publication_heading_is_recognised():
    from screener.extract import detect_heading

    assert detect_heading("Publication") == "publications"
