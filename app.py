"""File to execute the app"""

from app_files.multipage import MultiPage
app = MultiPage()

try:
    app.run()
except ValueError:
    app.run()
