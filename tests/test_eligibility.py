from screener.eligibility import check_eligibility
from screener.extract import split_sections
from screener.matching import compile_term


def elig(text, cfg):
    return check_eligibility(split_sections(text), cfg)


def test_java_react_only_profile_is_rejected_with_both_reasons(cfg, resume_text):
    result = elig(resume_text("java_react_only"), cfg)
    assert not result.eligible
    assert any("Python" in r for r in result.rejection_reasons)
    assert any("AI" in r for r in result.rejection_reasons)


def test_javascript_does_not_cause_rejection_when_python_and_ai_present(cfg, resume_text):
    result = elig(resume_text("fullstack_js_python_ai"), cfg)
    assert result.eligible
    assert result.ai_level == "llm_agentic"


def test_python_without_any_ai_is_rejected(cfg):
    text = "Sam Lee\nSKILLS\nPython, Django, PostgreSQL\nPROJECTS\nBlog API\n- Django REST API with PostgreSQL"
    result = elig(text, cfg)
    assert not result.eligible
    assert result.rejection_reasons == ["No AI/agentic project evidence"]


def test_ai_degree_name_and_certificates_are_not_evidence(cfg):
    text = ("Ravi K\nSKILLS\nJava, React\nEDUCATION\nB.Tech CSE (AI & ML specialisation)\n"
            "CERTIFICATIONS\nPython for Everybody; LangChain for LLM Apps (DeepLearning.AI)")
    result = elig(text, cfg)
    assert not result.eligible
    assert "only in education/certification" in result.rejection_reasons[0]


def test_python_implied_by_python_only_framework(cfg):
    text = "Mia\nPROJECTS\nDocs Bot\n- FastAPI service with LangChain RAG over PDFs"
    assert elig(text, cfg).eligible


def test_classical_ml_passes_gate_unless_disabled(cfg, resume_text):
    text = resume_text("classical_ml")
    assert elig(text, cfg).ai_level == "classical_ml"
    assert elig(text, cfg).eligible
    strict = {**cfg, "eligibility": {**cfg["eligibility"], "allow_classical_ml_only": False}}
    assert not elig(text, strict).eligible


def test_term_matching_avoids_classic_false_positives():
    assert not compile_term("RAG").search("drag and drop UI")
    assert not compile_term("LLM").search("LL.M. (Master of Laws)")
    assert not compile_term("embedding").search("embedded systems")
    assert compile_term("tool calling").search("tool-calling agents")
    assert compile_term("LLM").search("fine-tuned LLMs")


def test_classical_ml_only_listed_as_a_skill_does_not_pass_the_gate(cfg):
    text = ("Sam Lee\nSKILLS\nPython, Machine Learning Basics\n"
            "PROJECTS\nTravel Booking Site\n- Flask web app with MySQL and login")
    result = elig(text, cfg)
    assert not result.eligible
    assert "only listed as a skill" in result.rejection_reasons[0]


def test_classical_ml_in_a_project_passes_the_gate(cfg):
    text = ("Sam Lee\nSKILLS\nPython\n"
            "PROJECTS\nSoil Classifier\n- Trained scikit-learn models on soil data to reach 85% accuracy")
    result = elig(text, cfg)
    assert result.eligible and result.ai_level == "classical_ml"


def test_applied_evidence_rule_can_be_switched_off_in_config(cfg):
    text = ("Sam Lee\nSKILLS\nPython, Machine Learning Basics\n"
            "PROJECTS\nTravel Booking Site\n- Flask web app with MySQL")
    relaxed = {**cfg, "eligibility": {**cfg["eligibility"], "classical_ml_requires_applied_evidence": False}}
    assert elig(text, relaxed).eligible


def test_typescript_only_llm_project_is_rejected_for_missing_python(cfg):
    text = ("Mo Ali\nSKILLS\nTypeScript, React, Node.js\nPROJECTS\nRAG Platform\n"
            "- Built a RAG platform with pgvector, OpenAI embeddings and tool calling in TypeScript")
    result = elig(text, cfg)
    assert not result.eligible and result.rejection_reasons == ["No evidence of Python stack"]
