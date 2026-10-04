import reflex as rx

# main.py holds the app (app = rx.App(...)); the search backend lives in src/coderet.
# One locked theme: dark, a single blue accent, slate neutrals, medium radius.
config = rx.Config(
    app_name="main",
    app_module_import="main",
    show_built_with_reflex=False,
    plugins=[
        rx.plugins.RadixThemesPlugin(
            theme=rx.theme(appearance="dark", accent_color="blue", gray_color="slate", radius="medium",
                           panel_background="solid", scaling="100%")),
        rx.plugins.SitemapPlugin(),
    ],
)
