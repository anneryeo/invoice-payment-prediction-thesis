from dash import html, dcc

def UploadForm(upload_id, label):
    return html.Div(
        className="upload-area",
        children=[
            html.Label(label),
            dcc.Upload(
                id=upload_id,
                children=html.Div(["Drag and Drop or Select File"]),
                multiple=False
            )
        ]
    )