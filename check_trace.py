
from app import app

with app.test_client() as client:
    response = client.get("/")

    print("STATUS:", response.status_code)
    print("LIGHTBOX:", b'lightbox' in response.data)
    print("HTML SIZE:", len(response.data))
    print("\nHTML PREVIEW:")
    print(response.data[:300].decode("utf-8", errors="replace"))
    print("APP ROOT:", app.root_path)
    print("TEMPLATE:", app.jinja_env.get_template("index.html").filename)
