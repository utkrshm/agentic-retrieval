"""Reflex demo UI: paste a GitHub link, the repository is opened, then ask questions about its code.

    uv run reflex run --env prod --single-port

Backend: coderet.demo.Engine. Repositories that were indexed ahead of time (BrowserOS, Node-RED) open from their stored
indexes; any other public GitHub repository is cloned and indexed on the CPU. Search is the Gemma, jina-code and BM25
fusion on CPU.
"""

from __future__ import annotations

import asyncio
from typing import TypedDict

import reflex as rx

from coderet.demo import Engine

engine = Engine()
FONTS = ("https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600;700&"
         "family=Geist+Mono:wght@400;500&display=swap")
MONO = "var(--font-mono)"
LINE = "1px solid var(--gray-a5)"
PANEL = "var(--gray-a2)"
R_BOX = "12px"  # shape lock: containers 12px, controls follow the theme radius
PAD = "24px"  # one padding for every container; sections are separated by one gap
GAP = "32px"
PAGE_W = "90%"  # the whole page
RESULTS_W = "80%"  # search results: the central 80% of the page width


class Card(TypedDict):
    rank: int
    path: str
    dir: str
    file: str
    start: int
    end: int
    kind: str
    name: str
    lang: str
    code: str
    hidden: int
    url: str


class State(rx.State):
    link: str = ""
    repo_key: str = ""
    repo_label: str = ""
    repo_commit: str = ""
    repo_units: str = ""
    ready: bool = False
    loading: bool = False
    opening: bool = False
    open_message: str = ""
    open_error: str = ""
    query: str = ""
    tests_mode: str = "auto"
    n_results: str = "10"
    searching: bool = False
    searched: bool = False
    error: str = ""
    results: list[Card] = []
    asked: str = ""
    t_total: str = ""

    # ---- inputs ----------------------------------------------------------------------------------
    @rx.event
    def set_link(self, value: str):
        self.link = value

    @rx.event
    def set_query(self, value: str):
        self.query = value

    @rx.event
    def set_tests_mode(self, value: str):
        self.tests_mode = value

    @rx.event
    def set_n_results(self, value: str):
        self.n_results = value

    @rx.event
    def link_key_down(self, key: str):
        if key == "Enter":
            return State.open_repo

    @rx.event
    def query_key_down(self, key: str):
        if key == "Enter":
            return State.run_search

    # ---- models and repository -------------------------------------------------------------------
    @rx.event(background=True)
    async def load_engine(self):
        async with self:
            if self.ready or self.loading:
                return
            self.loading = True
        await asyncio.to_thread(engine.load_models)
        async with self:
            self.ready, self.loading = True, False

    @rx.event(background=True)
    async def open_repo(self):
        async with self:
            link = self.link.strip()
            if not link or self.opening:
                return
            self.opening, self.open_error, self.open_message = True, "", "Loading models" if not self.ready else "Opening the repository"
        while not engine.ready:  # the models load once at page open
            await asyncio.sleep(0.3)
        task = asyncio.create_task(asyncio.to_thread(engine.index, link))
        while not task.done():
            await asyncio.sleep(0.4)
            async with self:
                self.open_message = engine.progress.message
        try:
            repo = task.result()
        except Exception as exc:  # shown in the UI, never crash the demo
            async with self:
                self.opening = False
                self.open_error = "Cancelled." if type(exc).__name__ == "IndexingCancelled" else str(exc)
            return
        async with self:
            self.repo_key, self.repo_label, self.repo_commit = repo.key, repo.label, repo.commit[:7]
            self.repo_units = f"{repo.units:,}"
            self.opening, self.query, self.results, self.searched, self.error = False, "", [], False, ""

    @rx.event
    def cancel_open(self):
        engine.cancel()

    @rx.event
    def change_repo(self):
        self.repo_key, self.link, self.query, self.results, self.searched, self.error = "", "", "", [], False, ""

    # ---- search ----------------------------------------------------------------------------------
    @rx.event(background=True)
    async def run_search(self):
        async with self:
            query, key, mode, k = self.query.strip(), self.repo_key, self.tests_mode, int(self.n_results)
            if not query or not key or self.searching:
                return
            self.searching, self.error = True, ""
        try:
            out = await asyncio.to_thread(engine.search, key, query, mode, k)
        except Exception as exc:
            async with self:
                self.searching, self.error = False, f"{type(exc).__name__}: {exc}"
            return
        async with self:
            self.results, self.asked, self.searched = out["cards"], query, True
            self.t_total = str(out["timings"]["total"])
            self.searching = False


# ---- components ----------------------------------------------------------------------------------
def panel(*children: rx.Component, **props) -> rx.Component:
    return rx.vstack(*children, spacing="4", width="100%", padding=PAD, border=LINE, border_radius=R_BOX,
                     background=PANEL, align="start", **props)


def label(text: str) -> rx.Component:
    return rx.text(text, size="3", weight="medium", color="var(--gray-11)")


def title_bar() -> rx.Component:
    return rx.center(rx.heading("Agentic Code Retrieval", size="8", weight="bold", letter_spacing="-0.03em", line_height="1.1",
                                text_align="center"), width="100%")


def link_form() -> rx.Component:
    return panel(
        label("Repository"),
        rx.hstack(
            rx.input(rx.input.slot(rx.icon("git-branch", size=20)), placeholder="https://github.com/owner/repository",
                     value=State.link, on_change=State.set_link, on_key_down=State.link_key_down, size="3",
                     width="100%", disabled=State.opening, font_size="16px"),
            rx.button(rx.cond(State.opening, rx.spinner(size="2"), rx.icon("arrow-right", size=18)),
                      rx.cond(State.opening, "Opening", "Open repository"), class_name="search-button",
                      on_click=State.open_repo, size="3", disabled=State.opening, flex_shrink="0"),
            width="100%", spacing="3"),
        rx.cond(State.opening,
                rx.hstack(rx.spinner(size="2"), rx.text(State.open_message, size="3", color="var(--gray-11)"),
                          rx.button("Cancel", on_click=State.cancel_open, variant="ghost", size="2", color_scheme="gray"),
                          spacing="3", align="center")),
        rx.cond(State.open_error != "", rx.callout(State.open_error, icon="triangle-alert", color_scheme="red", width="100%")))


def repo_bar() -> rx.Component:
    return panel(
        rx.hstack(
            rx.icon("git-branch", size=22, color="var(--gray-11)"),
            rx.text(State.repo_label, size="4", weight="bold"),
            rx.text("commit " + State.repo_commit, size="3", color="var(--gray-11)", font_family=MONO),
            rx.text(State.repo_units + " units", size="3", color="var(--gray-11)", font_family=MONO),
            rx.spacer(),
            rx.button("Change repository", on_click=State.change_repo, variant="soft", color_scheme="gray", size="2"),
            width="100%", align="center", spacing="4", wrap="wrap"))


def question_panel() -> rx.Component:
    return panel(
        label("Question"),
        rx.hstack(
            rx.input(rx.input.slot(rx.icon("search", size=20)), placeholder="Where is the retry logic for failed tool calls?",
                     value=State.query, on_change=State.set_query, on_key_down=State.query_key_down, size="3",
                     width="100%", font_size="16px", auto_focus=True),
            rx.button(rx.cond(State.searching, rx.spinner(size="2"), rx.icon("arrow-right", size=18)), "Search",
                      class_name="search-button", on_click=State.run_search, size="3",
                      disabled=State.searching | ~State.ready, flex_shrink="0"),
            width="100%", spacing="3"),
        rx.hstack(label("Test files"),
                  rx.select(["auto", "include", "exclude"], value=State.tests_mode, on_change=State.set_tests_mode, size="2"),
                  rx.box(width="16px"),
                  label("Results"),
                  rx.select(["1", "5", "10"], value=State.n_results, on_change=State.set_n_results, size="2"),
                  spacing="3", align="center", wrap="wrap"))


def result_card(r: rx.Var) -> rx.Component:
    return rx.vstack(
        rx.hstack(
            rx.text(r["rank"], size="6", weight="bold", font_family=MONO, color="var(--gray-9)", min_width="28px"),
            rx.vstack(
                rx.hstack(
                    rx.text(r["dir"], size="2", color="var(--gray-10)", font_family=MONO, word_break="break-all"),
                    rx.text(r["file"], size="2", weight="bold", font_family=MONO),
                    spacing="0", align="baseline", wrap="wrap"),
                rx.hstack(
                    rx.badge(r["kind"], variant="soft", color_scheme="gray", size="1"),
                    rx.text(r["name"], size="2", color="var(--gray-12)", font_family=MONO),
                    rx.text("lines " + r["start"].to_string() + " to " + r["end"].to_string(), size="1", color="var(--gray-10)"),
                    spacing="2", align="center", wrap="wrap"),
                spacing="1", align="start", min_width="0"),
            rx.spacer(),
            rx.link(rx.hstack(rx.text("GitHub", size="2"), rx.icon("arrow-up-right", size=15), spacing="1", align="center"),
                    href=r["url"], is_external=True, color="var(--gray-11)", underline="none", flex_shrink="0"),
            width="100%", align="start", spacing="3"),
        rx.box(rx.code_block(r["code"], language=r["lang"], show_line_numbers=True, starting_line_number=r["start"],
                             width="100%", font_size="12px", wrap_long_lines=False),
               width="100%", overflow_x="auto", border_radius="8px"),
        rx.cond(r["hidden"] > 0, rx.text("+ " + r["hidden"].to_string() + " more lines in this unit", size="1",
                                         color="var(--gray-10)")),
        class_name="result-card", style={"--i": r["rank"]}, spacing="4", width="100%", min_width="0", padding=PAD,
        border=LINE, border_radius=R_BOX, background=PANEL, align="start",
        border_left=rx.cond(r["rank"] == 1, "3px solid var(--accent-9)", LINE))


def results_view() -> rx.Component:
    return rx.vstack(
        rx.cond(State.error != "", rx.callout(State.error, icon="triangle-alert", color_scheme="red", width="100%")),
        rx.cond(
            State.searching,
            rx.vstack(*[rx.skeleton(height="220px", width="100%", border_radius=R_BOX) for _ in range(3)],
                      spacing="5", width="100%"),
            rx.cond(
                State.searched,
                rx.vstack(
                    rx.hstack(
                        rx.hstack(rx.text("Top ", size="4", color="var(--gray-11)"),
                                  rx.text(State.results.length().to_string(), size="4", weight="bold"),
                                  rx.text(" matches for ", size="4", color="var(--gray-11)"),
                                  rx.text("\"" + State.asked + "\"", size="4", weight="medium"), wrap="wrap", spacing="1"),
                        rx.spacer(),
                        rx.hstack(rx.text("Retrieved in", size="3", color="var(--gray-11)"),
                                  rx.text(State.t_total, size="5", weight="bold", font_family=MONO, color="var(--accent-11)"),
                                  rx.text("ms", size="3", color="var(--accent-11)"), spacing="2", align="baseline"),
                        width="100%", align="baseline", spacing="4", wrap="wrap", padding_y="3px"),
                    rx.cond(State.results.length() == 0,
                            rx.text("No matches. Try naming a function, a setting or an error message.", size="3",
                                    color="var(--gray-11)")),
                    rx.vstack(rx.foreach(State.results, result_card), spacing="5", width="100%"),
                    spacing="4", width="100%"))),
        spacing="4", width=RESULTS_W, margin="0 auto", align="start")


def footer() -> rx.Component:
    return rx.box(
        rx.text("Gemma retrieves the top 50, jina-code and BM25 re-score them. Exact search, no GPU at query time.",
                size="2", color="var(--gray-10)"),
        width="100%", padding_y=PAD, border_top=LINE, margin_top="auto")


def index() -> rx.Component:
    return rx.box(
        rx.vstack(
            title_bar(),
            rx.cond(State.repo_key == "", link_form(), rx.vstack(repo_bar(), question_panel(), results_view(),
                                                                  spacing="5", width="100%")),
            spacing="6", width="100%", padding_top="40px", padding_bottom=GAP, align="start"),
        footer(),
        display="flex", flex_direction="column", width=PAGE_W, margin="0 auto", min_height="100dvh")


app = rx.App(
    stylesheets=[FONTS, "/app.css"],
    style={"font_family": "var(--font-sans)"},
)
app.add_page(index, route="/", title="Agentic Code Retrieval", on_load=State.load_engine)


async def preload_models():
    """Load both models when the backend starts so the first page open is already ready."""
    await asyncio.to_thread(engine.load_models)


app.register_lifespan_task(preload_models)
