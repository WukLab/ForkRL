"""WebArena-Lite scoring for the 45 shopping tasks, ported from VAB-WebArena-Lite (THUDM VisualAgentBench @9055fc2)
without the visualwebarena package: string_match, url_match, program_html, and the one helper they use. Differences:
the page is our Playwright page inside the environment (__SHOPPING__ = http://localhost), the order helper calls
Magento's REST API from that page, and fuzzy_match's judge is any OpenAI-compatible endpoint (none: unscored).

Entry point: await evaluate(task, answer, page, judge=None) -> 1.0 / 0.0, or None when unscored;
load_tasks(path=None) -> {task_id: task}.

Notes: a single-word must_include reference must match a whole token ("0" is not in "10"); program_html keeps the
  original's 3 s sleep after a navigation.
"""
import asyncio, json

from nltk.tokenize import word_tokenize

SHOPPING = "http://localhost"
ADMIN = ("admin", "admin1234")


class Unscored(Exception):
    pass


def clean_answer(answer):
    answer = str(answer)
    if answer.startswith("'") and answer.endswith("'"):
        answer = answer[1:-1]
    elif answer.startswith('"') and answer.endswith('"'):
        answer = answer[1:-1]
    return answer.lower()


def exact_match(ref, pred):
    return float(clean_answer(pred) == clean_answer(ref))


def must_include(ref, pred):
    clean_ref, clean_pred = clean_answer(ref), clean_answer(pred)
    if len(word_tokenize(clean_ref)) == 1:
        tok_pred = word_tokenize(clean_pred)
        for token in list(tok_pred):
            if "/" in token:
                tok_pred.extend(token.split("/"))
        return float(clean_ref in tok_pred)
    return float(clean_ref in clean_pred)


FUZZY_PROMPT = ("Help a teacher to grade the answer of a student given a question. Keep in mind that the student may "
                "use different phrasing or wording to answer the question. The goal is to evaluate whether the answer "
                "is semantically equivalent to the reference answer.\n"
                "question: {question}\nreference answer: {reference}\n"
                "all the string 'N/A' that you see is a special sequence that means 'not achievable'\n"
                "student answer: {pred}\n"
                "Conclude the judgement by 'correct', 'incorrect', or 'partially correct'. Only output one of these "
                "options, and nothing else.")


async def fuzzy_match(ref, pred, intent, judge):
    if judge is None:
        raise Unscored("fuzzy_match needs a judge")
    client, model = judge
    r = await client.chat.completions.create(model=model, temperature=0, max_tokens=768, top_p=1.0, messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": FUZZY_PROMPT.format(question=intent, reference=ref, pred=pred)}])
    response = (r.choices[0].message.content or "").lower()
    if "partially correct" in response or "incorrect" in response:
        return 0.0
    if "correct" not in response:
        raise ValueError(f"judge gave no verdict: {response[:200]!r}")
    return 1.0


async def string_match(task, answer, judge):
    pred, score = clean_answer(answer), 1.0
    for approach, value in task["eval"]["reference_answers"].items():
        if approach == "exact_match":
            score *= exact_match(value, pred)
        elif approach == "must_include":
            for must_value in value:
                score *= any(must_include(v, pred) for v in must_value.split(" |OR| "))
        elif approach == "fuzzy_match":
            if value == "N/A":
                raise NotImplementedError("fuzzy_match N/A")
            score *= await fuzzy_match(", ".join(value), pred, task["intent"], judge)
        else:
            raise NotImplementedError(approach)
    return score


def clean_url(url):
    url = str(url).replace("localhost", "127.0.0.1")
    return url[:-1] if url.endswith("/") else url


def url_match(task, page_url):
    pred = clean_url(page_url)
    refs = [clean_url(u.replace("__SHOPPING__", SHOPPING)) for u in task["eval"]["reference_url"].split(" |OR| ")]
    rule = task["eval"].get("url_note", "EXACT")
    if rule == "EXACT":
        return float(pred in refs)
    if rule == "GOLD in PRED":
        return float(any(ref in pred for ref in refs))
    raise ValueError(rule)


LATEST_ORDER_JS = """async ([user, pw]) => {
  const t = await (await fetch('/rest/default/V1/integration/admin/token', {method: 'POST',
      headers: {'content-type': 'application/json'}, body: JSON.stringify({username: user, password: pw})})).json();
  const q = 'searchCriteria[sortOrders][0][field]=created_at&searchCriteria[sortOrders][0][direction]=DESC'
          + '&searchCriteria[pageSize]=1';
  const r = await fetch('/rest/V1/orders?' + q, {headers: {'Authorization': 'Bearer ' + t, 'Content-Type': 'application/json'}});
  if (r.status !== 200) throw new Error('orders: http ' + r.status);
  return (await r.json()).items[0].increment_id;
}"""


async def shopping_get_latest_order_url(page):
    order_id = int(await page.evaluate(LATEST_ORDER_JS, list(ADMIN)))
    return f"{SHOPPING}/sales/order/view/order_id/{order_id}/"


async def program_html(task, page, judge):
    score = 1.0
    for target in task["eval"]["program_html"]:
        target_url = target["url"]
        if target_url.startswith("func:"):
            func = target_url.split("func:")[1]
            if func != "shopping_get_latest_order_url()":
                raise NotImplementedError(func)
            target_url = await shopping_get_latest_order_url(page)
        else:
            target_url = target_url.replace("__SHOPPING__", SHOPPING)
        locator = target["locator"]
        if target_url != "last":
            await page.goto(target_url)
            await asyncio.sleep(3)
        if not locator.strip():
            selected = await page.content()
        elif locator.startswith("document.") or locator.startswith("[...document."):
            for prep in target.get("prep_actions", []):
                try:
                    await page.evaluate(f"() => {prep}")
                except Exception:
                    pass
            try:
                selected = str(await page.evaluate(f"() => {locator}")) or ""
            except Exception:
                selected = ""
        else:
            raise NotImplementedError(locator)
        req = target["required_contents"]
        if "exact_match" in req:
            score *= exact_match(req["exact_match"], selected)
        elif "must_include" in req:
            for content in req["must_include"]:
                score *= any(must_include(c, selected) for c in content.split(" |OR| "))
        elif "fuzzy_match" in req:
            score *= await fuzzy_match(", ".join(req["fuzzy_match"]), selected, task["intent"], judge)
        else:
            raise NotImplementedError(list(req))
    return score


async def evaluate(task, answer, page, judge=None):
    score = 1.0
    try:
        for kind in task["eval"]["eval_types"]:
            if kind == "string_match":
                score *= await string_match(task, answer, judge)
            elif kind == "url_match":
                score *= url_match(task, page.url)
            elif kind == "program_html":
                score *= await program_html(task, page, judge)
            else:
                raise NotImplementedError(kind)
    except Unscored:
        return None
    return float(score)


def load_tasks(path=None):
    from pathlib import Path
    return {t["task_id"]: t for t in json.load(open(path or Path(__file__).with_name("tasks") / "lite_shopping.json"))}
