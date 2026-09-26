from flask import Flask, render_template, request, send_from_directory, redirect, session, flash, jsonify
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2
from werkzeug.security import generate_password_hash, check_password_hash
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer, util
import sqlite3
import os
import uuid
import json
import math
import urllib.request
import urllib.parse
import torch
import torchvision.models as models
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE = os.path.join(BASE_DIR, "lost_found.db")

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "lost-found-secret-key")
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

text_model = SentenceTransformer("all-MiniLM-L6-v2")

image_model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)
image_model.fc = torch.nn.Identity()
image_model.eval()

image_transform = models.ResNet18_Weights.DEFAULT.transforms()

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

NO_CACHE_ENDPOINTS = {
    "report_lost",
    "report_found"
}


def init_db():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS lost_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            item_name TEXT,
            description TEXT,
            location TEXT,
            date TEXT,
            image TEXT,
            latitude REAL,
            longitude REAL,
            category TEXT,
            user_id INTEGER,
            status TEXT DEFAULT 'ACTIVE',
            deleted INTEGER DEFAULT 0
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS found_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            item_name TEXT,
            description TEXT,
            location TEXT,
            date TEXT,
            image TEXT,
            latitude REAL,
            longitude REAL,
            category TEXT,
            user_id INTEGER,
            status TEXT DEFAULT 'ACTIVE',
            deleted INTEGER DEFAULT 0
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT UNIQUE,
            password TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS matches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lost_item_id INTEGER,
            found_item_id INTEGER,
            score REAL,
            reasons TEXT,
            status TEXT DEFAULT 'ACTIVE',
            lost_confirmed INTEGER DEFAULT 0,
            found_confirmed INTEGER DEFAULT 0,
            name_similarity REAL,
            description_similarity REAL,
            image_similarity REAL,
            UNIQUE(lost_item_id, found_item_id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS notifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            match_id INTEGER,
            message TEXT,
            is_read INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS item_embeddings (
            item_type TEXT,
            item_id INTEGER,
            name_embedding TEXT,
            description_embedding TEXT,
            image_embedding TEXT,
            PRIMARY KEY (item_type, item_id)
        )
    """)

    for table in ["lost_items", "found_items"]:
        columns = [
            row[1]
            for row in cursor.execute(
                f"PRAGMA table_info({table})"
            ).fetchall()
        ]

        if "image" not in columns:
            cursor.execute(
                f"ALTER TABLE {table} ADD COLUMN image TEXT"
            )

        if "latitude" not in columns:
            cursor.execute(
                f"ALTER TABLE {table} ADD COLUMN latitude REAL"
            )

        if "longitude" not in columns:
            cursor.execute(
                f"ALTER TABLE {table} ADD COLUMN longitude REAL"
            )

        if "category" not in columns:
            cursor.execute(
                f"ALTER TABLE {table} ADD COLUMN category TEXT"
            )

        if "user_id" not in columns:
            cursor.execute(
                f"ALTER TABLE {table} ADD COLUMN user_id INTEGER"
            )

        if "status" not in columns:
            cursor.execute(
                f"ALTER TABLE {table} ADD COLUMN status TEXT DEFAULT 'ACTIVE'"
            )

        if "deleted" not in columns:
            cursor.execute(
                f"ALTER TABLE {table} ADD COLUMN deleted INTEGER DEFAULT 0"
            )

    match_columns = [
        row[1]
        for row in cursor.execute(
            "PRAGMA table_info(matches)"
        ).fetchall()
    ]

    if "lost_confirmed" not in match_columns:
        cursor.execute(
            "ALTER TABLE matches ADD COLUMN lost_confirmed INTEGER DEFAULT 0"
        )

    if "found_confirmed" not in match_columns:
        cursor.execute(
            "ALTER TABLE matches ADD COLUMN found_confirmed INTEGER DEFAULT 0"
        )

    if "name_similarity" not in match_columns:
        cursor.execute(
            "ALTER TABLE matches ADD COLUMN name_similarity REAL"
        )

    if "description_similarity" not in match_columns:
        cursor.execute(
            "ALTER TABLE matches ADD COLUMN description_similarity REAL"
        )

    if "image_similarity" not in match_columns:
        cursor.execute(
            "ALTER TABLE matches ADD COLUMN image_similarity REAL"
        )

    conn.commit()
    conn.close()


@app.after_request
def add_no_cache_headers(response):
    if request.endpoint in NO_CACHE_ENDPOINTS:
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"

    return response


@app.context_processor
def global_user_data():
    unread_count = 0

    if "user_id" in session:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT COUNT(*)
            FROM notifications
            WHERE user_id = ?
            AND is_read = 0
        """, (session["user_id"],))

        unread_count = cursor.fetchone()[0]
        conn.close()

    return {
        "logged_in": "user_id" in session,
        "current_user_name": session.get("user_name"),
        "unread_notification_count": unread_count
    }


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not name or not email or not password:
            return "All fields are required."

        if len(password) < 6:
            return "Password must contain at least 6 characters."

        hashed_password = generate_password_hash(password)

        conn = sqlite3.connect(DATABASE)

        try:
            conn.execute("""
                INSERT INTO users
                (name, email, password)
                VALUES (?, ?, ?)
            """, (
                name,
                email,
                hashed_password
            ))

            conn.commit()

        except sqlite3.IntegrityError:
            conn.close()
            return "Email already registered."

        conn.close()

        flash(
            "Account created successfully. Please login.",
            "success"
        )

        return redirect("/login")

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM users WHERE email = ?",
            (email,)
        )

        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user[3], password):
            session["user_id"] = user[0]
            session["user_name"] = user[1]

            return redirect("/dashboard")

        return "Invalid email or password."

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")


@app.route("/")
def home():
    return render_template("home.html")


def save_uploaded_image(image):
    if not image or not image.filename:
        return None

    extension = os.path.splitext(
        image.filename
    )[1].lower()

    allowed_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp"
    }

    if extension not in allowed_extensions:
        return None

    filename = f"{uuid.uuid4()}{extension}"

    path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    try:
        image.save(path)

        with Image.open(path) as uploaded:
            uploaded.verify()

    except Exception:
        if os.path.exists(path):
            os.remove(path)

        return None

    return filename


@app.route("/api/geocode")
def api_geocode():
    if "user_id" not in session:
        return jsonify({
            "error": "unauthorized"
        }), 401

    query = request.args.get(
        "q",
        ""
    ).strip()

    if not query:
        return jsonify({
            "results": []
        })

    url = "https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode({
        "q": query,
        "format": "json",
        "limit": 5
    })

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "lost-found-smart-recovery/1.0"
        }
    )

    try:
        with urllib.request.urlopen(
            req,
            timeout=5
        ) as response:
            data = json.loads(
                response.read().decode()
            )

    except Exception:
        return jsonify({
            "results": []
        })

    results = [
        {
            "display_name": item.get("display_name"),
            "latitude": item.get("lat"),
            "longitude": item.get("lon")
        }
        for item in data
    ]

    return jsonify({
        "results": results
    })


def create_item(table, item_type):
    if "user_id" not in session:
        return redirect("/login")

    if request.method == "POST":
        item_name = request.form.get(
            "item_name",
            ""
        ).strip()

        description = request.form.get(
            "description",
            ""
        ).strip()

        location = request.form.get(
            "location",
            ""
        ).strip()

        date = request.form.get(
            "date",
            ""
        ).strip()

        category = request.form.get(
            "category",
            ""
        ).strip()

        latitude = request.form.get(
            "latitude"
        ) or None

        longitude = request.form.get(
            "longitude"
        ) or None

        if (
            not item_name
            or not description
            or not location
            or not date
        ):
            flash(
                "Please fill all required fields.",
                "error"
            )

            if item_type == "lost":
                return render_template(
                    "report_lost.html"
                )

            return render_template(
                "found_item.html"
            )

        image = save_uploaded_image(
            request.files.get("image")
        )

        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()

        cursor.execute(
            f"""
            INSERT INTO {table}
            (
                item_name,
                description,
                location,
                date,
                image,
                latitude,
                longitude,
                category,
                user_id,
                status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item_name,
                description,
                location,
                date,
                image,
                latitude,
                longitude,
                category,
                session["user_id"],
                "ACTIVE"
            )
        )

        new_id = cursor.lastrowid

        conn.commit()
        conn.close()

        try:
            process_new_item(
                item_type,
                new_id
            )
        except Exception:
            pass

        if item_type == "lost":
            flash(
                "Lost item reported successfully!",
                "success"
            )

            return redirect(
                "/lost-items"
            )

        flash(
            "Found item reported successfully!",
            "success"
        )

        return redirect(
            "/found-items"
        )

    if item_type == "lost":
        return render_template(
            "report_lost.html"
        )

    return render_template(
        "found_item.html"
    )


@app.route(
    "/report-lost",
    methods=["GET", "POST"]
)
def report_lost():
    return create_item(
        "lost_items",
        "lost"
    )


@app.route(
    "/report-found",
    methods=["GET", "POST"]
)
def report_found():
    return create_item(
        "found_items",
        "found"
    )


def get_items(
    table,
    search="",
    category="",
    location=""
):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    query = f"""
        SELECT *
        FROM {table}
        WHERE status = 'ACTIVE'
        AND deleted = 0
    """

    parameters = []

    if search:
        query += """
            AND (
                LOWER(item_name) LIKE LOWER(?)
                OR LOWER(description) LIKE LOWER(?)
                OR LOWER(location) LIKE LOWER(?)
            )
        """

        value = "%" + search + "%"

        parameters.extend([
            value,
            value,
            value
        ])

    if category:
        query += """
            AND LOWER(category) = LOWER(?)
        """

        parameters.append(
            category
        )

    if location:
        query += """
            AND LOWER(location) LIKE LOWER(?)
        """

        parameters.append(
            "%" + location + "%"
        )

    query += " ORDER BY id DESC"

    cursor.execute(
        query,
        parameters
    )

    items = cursor.fetchall()

    conn.close()

    return items


@app.route("/lost-items")
def lost_items():
    if "user_id" not in session:
        return redirect("/login")

    search = request.args.get(
        "search",
        ""
    ).strip()

    category = request.args.get(
        "category",
        ""
    ).strip()

    location = request.args.get(
        "location",
        ""
    ).strip()

    items = get_items(
        "lost_items",
        search,
        category,
        location
    )

    return render_template(
        "lost_items.html",
        items=items,
        search=search,
        category=category,
        location=location
    )


@app.route("/found-items")
def view_found_items():
    if "user_id" not in session:
        return redirect("/login")

    search = request.args.get(
        "search",
        ""
    ).strip()

    category = request.args.get(
        "category",
        ""
    ).strip()

    location = request.args.get(
        "location",
        ""
    ).strip()

    items = get_items(
        "found_items",
        search,
        category,
        location
    )

    return render_template(
        "found_items.html",
        items=items,
        search=search,
        category=category,
        location=location
    )


@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect("/login")

    user_id = session["user_id"]

    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT COUNT(*)
        FROM lost_items
        WHERE status = 'ACTIVE'
        AND deleted = 0
        AND user_id = ?
    """, (user_id,))

    lost_count = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*)
        FROM found_items
        WHERE status = 'ACTIVE'
        AND deleted = 0
        AND user_id = ?
    """, (user_id,))

    found_count = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*)
        FROM matches m
        JOIN lost_items l
            ON m.lost_item_id = l.id
        JOIN found_items f
            ON m.found_item_id = f.id
        WHERE m.status = 'ACTIVE'
        AND l.status = 'ACTIVE'
        AND f.status = 'ACTIVE'
        AND l.deleted = 0
        AND f.deleted = 0
        AND (
            l.user_id = ?
            OR f.user_id = ?
        )
    """, (
        user_id,
        user_id
    ))

    match_count = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*)
        FROM notifications
        WHERE user_id = ?
        AND is_read = 0
    """, (user_id,))

    notification_count = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*)
        FROM lost_items
        WHERE user_id = ?
        AND status = 'RESOLVED'
    """, (user_id,))

    resolved_lost = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*)
        FROM found_items
        WHERE user_id = ?
        AND status = 'RESOLVED'
    """, (user_id,))

    resolved_found = cursor.fetchone()[0]

    conn.close()

    return render_template(
        "dashboard.html",
        lost_count=lost_count,
        found_count=found_count,
        match_count=match_count,
        notification_count=notification_count,
        resolved_lost=resolved_lost,
        resolved_found=resolved_found
    )


@app.route("/my-reports")
def my_reports():
    if "user_id" not in session:
        return redirect("/login")

    user_id = session["user_id"]

    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            id,
            item_name,
            description,
            location,
            date,
            image,
            category,
            status,
            'LOST' AS item_type
        FROM lost_items
        WHERE user_id = ?
        AND deleted = 0

        UNION ALL

        SELECT
            id,
            item_name,
            description,
            location,
            date,
            image,
            category,
            status,
            'FOUND' AS item_type
        FROM found_items
        WHERE user_id = ?
        AND deleted = 0

        ORDER BY id DESC
    """, (
        user_id,
        user_id
    ))

    reports = cursor.fetchall()

    conn.close()

    return render_template(
        "my_reports.html",
        reports=reports
    )


def normalize_text(text):
    if text is None:
        return ""

    text = str(text).lower()

    for character in ",.!?;:-_()[]{}":
        text = text.replace(
            character,
            " "
        )

    stop_words = {
        "the",
        "a",
        "an",
        "is",
        "was",
        "in",
        "on",
        "at",
        "of",
        "and",
        "or",
        "to",
        "with"
    }

    words = [
        word
        for word in text.split()
        if word not in stop_words
    ]

    return " ".join(words)


def cosine_similarity(
    vector1,
    vector2
):
    if not vector1 or not vector2:
        return 0

    try:
        dot = sum(
            a * b
            for a, b in zip(
                vector1,
                vector2
            )
        )

        norm1 = math.sqrt(
            sum(
                a * a
                for a in vector1
            )
        )

        norm2 = math.sqrt(
            sum(
                b * b
                for b in vector2
            )
        )

        if norm1 == 0 or norm2 == 0:
            return 0

        similarity = dot / (
            norm1 * norm2
        )

        return max(
            0,
            min(1, similarity)
        )

    except (
        TypeError,
        ValueError,
        ZeroDivisionError
    ):
        return 0


def text_similarity_from_vectors(
    text1,
    text2,
    vector1,
    vector2
):
    normalized1 = normalize_text(text1)
    normalized2 = normalize_text(text2)

    if not normalized1 or not normalized2:
        return {
            "word": 0,
            "character": 0,
            "semantic": 0
        }

    words1 = set(
        normalized1.split()
    )

    words2 = set(
        normalized2.split()
    )

    common = words1 & words2

    word_similarity = len(common) / max(
        len(words1),
        len(words2)
    )

    character_similarity = SequenceMatcher(
        None,
        normalized1,
        normalized2
    ).ratio()

    if vector1 is not None and vector2 is not None:
        semantic_similarity = cosine_similarity(
            vector1,
            vector2
        )
    else:
        try:
            embeddings = text_model.encode(
                [
                    normalized1,
                    normalized2
                ],
                convert_to_tensor=True
            )

            semantic_similarity = float(
                util.cos_sim(
                    embeddings[0],
                    embeddings[1]
                )[0][0]
            )

        except Exception:
            semantic_similarity = 0

    semantic_similarity = max(
        0,
        min(
            1,
            semantic_similarity
        )
    )

    return {
        "word": word_similarity,
        "character": character_similarity,
        "semantic": semantic_similarity
    }


def combined_text_score(similarity):
    return (
        similarity["semantic"] * 0.60
        + similarity["character"] * 0.25
        + similarity["word"] * 0.15
    )


def location_similarity(
    text1,
    text2
):
    text1 = normalize_text(text1)
    text2 = normalize_text(text2)

    if not text1 or not text2:
        return 0

    words1 = set(
        text1.split()
    )

    words2 = set(
        text2.split()
    )

    common = words1 & words2

    word_similarity = len(common) / max(
        len(words1),
        len(words2)
    )

    character_similarity = SequenceMatcher(
        None,
        text1,
        text2
    ).ratio()

    return max(
        word_similarity,
        character_similarity
    )


def calculate_distance(
    lat1,
    lon1,
    lat2,
    lon2
):
    R = 6371

    lat1 = radians(float(lat1))
    lon1 = radians(float(lon1))
    lat2 = radians(float(lat2))
    lon2 = radians(float(lon2))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        sin(dlat / 2) ** 2
        + cos(lat1)
        * cos(lat2)
        * sin(dlon / 2) ** 2
    )

    a = min(
        1,
        max(
            0,
            a
        )
    )

    c = 2 * atan2(
        sqrt(a),
        sqrt(1 - a)
    )

    return R * c


def compute_image_vector(filename):
    if not filename:
        return None

    path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    if not os.path.exists(path):
        return None

    try:
        with Image.open(path) as image:
            image = image.convert("RGB")
            tensor = image_transform(
                image
            ).unsqueeze(0)

        with torch.no_grad():
            embedding = image_model(
                tensor
            )

        embedding = torch.nn.functional.normalize(
            embedding,
            p=2,
            dim=1
        )

        return embedding.squeeze().tolist()

    except Exception:
        return None


def store_embeddings(
    cursor,
    item_type,
    item_id,
    item_name,
    description,
    image_filename
):
    name_vector = None
    description_vector = None
    image_vector = None

    normalized_name = normalize_text(
        item_name
    )

    normalized_description = normalize_text(
        description
    )

    try:
        if normalized_name:
            name_vector = text_model.encode(
                normalized_name
            ).tolist()

        if normalized_description:
            description_vector = text_model.encode(
                normalized_description
            ).tolist()
    except Exception:
        name_vector = None
        description_vector = None

    if image_filename:
        image_vector = compute_image_vector(
            image_filename
        )

    cursor.execute("""
        INSERT OR REPLACE INTO item_embeddings
        (
            item_type,
            item_id,
            name_embedding,
            description_embedding,
            image_embedding
        )
        VALUES (?, ?, ?, ?, ?)
    """, (
        item_type,
        item_id,
        json.dumps(name_vector)
        if name_vector is not None
        else None,
        json.dumps(description_vector)
        if description_vector is not None
        else None,
        json.dumps(image_vector)
        if image_vector is not None
        else None
    ))

    return (
        name_vector,
        description_vector,
        image_vector
    )


def get_or_compute_embeddings(
    cursor,
    item_type,
    item_id,
    item_name,
    description,
    image_filename
):
    return store_embeddings(
        cursor,
        item_type,
        item_id,
        item_name,
        description,
        image_filename
    )


def compute_match(
    lost,
    found,
    lost_vectors,
    found_vectors
):
    lost_name_vec, lost_desc_vec, lost_img_vec = lost_vectors
    found_name_vec, found_desc_vec, found_img_vec = found_vectors

    lost_name = lost[1]
    found_name = found[1]

    lost_description = lost[2]
    found_description = found[2]

    lost_location = lost[3]
    found_location = found[3]

    name_similarity = text_similarity_from_vectors(
        lost_name,
        found_name,
        lost_name_vec,
        found_name_vec
    )

    description_similarity = text_similarity_from_vectors(
        lost_description,
        found_description,
        lost_desc_vec,
        found_desc_vec
    )

    name_score = combined_text_score(
        name_similarity
    )

    description_score = combined_text_score(
        description_similarity
    )

    visual_score = cosine_similarity(
        lost_img_vec,
        found_img_vec
    ) if (
        lost_img_vec
        and found_img_vec
    ) else 0

    score = 0
    reasons = []

    if (
        normalize_text(lost_name)
        and normalize_text(found_name)
    ):
        score += name_score * 20

        if normalize_text(lost_name) == normalize_text(found_name):
            reasons.append(
                "Item names match exactly"
            )
        elif name_score >= 0.75:
            reasons.append(
                "AI and text analysis found the item names highly similar"
            )
        elif name_score >= 0.5:
            reasons.append(
                "AI and text analysis found the item names similar"
            )

    if (
        normalize_text(lost_description)
        and normalize_text(found_description)
    ):
        score += description_score * 25

        if description_score >= 0.75:
            reasons.append(
                "AI and text analysis found the descriptions highly similar"
            )
        elif description_score >= 0.5:
            reasons.append(
                "AI and text analysis found the descriptions similar"
            )
        elif description_score >= 0.25:
            reasons.append(
                "Descriptions have some similarity"
            )

    if lost[5] and found[5]:
        score += visual_score * 20

        if visual_score >= 0.80:
            reasons.append(
                "AI image analysis found the images highly similar"
            )
        elif visual_score >= 0.60:
            reasons.append(
                "AI image analysis found the images similar"
            )
        elif visual_score >= 0.40:
            reasons.append(
                "Images have some visual similarity"
            )

    lost_category = normalize_text(
        lost[8]
    )

    found_category = normalize_text(
        found[8]
    )

    if (
        lost_category
        and found_category
        and lost_category == found_category
    ):
        score += 10

        reasons.append(
            "Items belong to the same category"
        )

    if lost_location and found_location:
        location_score = location_similarity(
            lost_location,
            found_location
        )

        score += location_score * 5

        if normalize_text(lost_location) == normalize_text(found_location):
            reasons.append(
                "Reported locations match"
            )
        elif location_score >= 0.75:
            reasons.append(
                "Reported locations are highly similar"
            )
        elif location_score >= 0.5:
            reasons.append(
                "Reported locations are similar"
            )

    date_difference = None

    if lost[4] and found[4]:
        try:
            lost_date = datetime.strptime(
                lost[4],
                "%Y-%m-%d"
            )

            found_date = datetime.strptime(
                found[4],
                "%Y-%m-%d"
            )

            date_difference = abs(
                (
                    lost_date
                    - found_date
                ).days
            )

        except ValueError:
            date_difference = None

    if date_difference == 0:
        score += 5

        reasons.append(
            "Same date"
        )

    elif date_difference == 1:
        score += 2.5

        reasons.append(
            "Dates are 1 day apart"
        )

    distance = None

    if (
        lost[6] is not None
        and lost[7] is not None
        and found[6] is not None
        and found[7] is not None
    ):
        try:
            distance = calculate_distance(
                lost[6],
                lost[7],
                found[6],
                found[7]
            )

        except (
            ValueError,
            TypeError
        ):
            distance = None

        if distance is not None:
            if distance <= 0.1:
                score += 15

                reasons.append(
                    "Locations are within 100 meters"
                )

            elif distance <= 0.5:
                score += 11

                reasons.append(
                    "Locations are within 500 meters"
                )

            elif distance <= 1:
                score += 7

                reasons.append(
                    "Locations are within 1 km"
                )

    score = round(
        min(
            score,
            100
        )
    )

    name_pct = round(
        name_similarity["semantic"] * 100
    )

    description_pct = round(
        description_similarity["semantic"] * 100
    )

    image_pct = round(
        visual_score * 100
    )

    return (
        score,
        reasons,
        distance,
        name_pct,
        description_pct,
        image_pct
    )


def upsert_match_record(
    cursor,
    lost,
    found,
    score,
    reasons,
    name_pct,
    description_pct,
    image_pct
):
    existing_match = cursor.execute("""
        SELECT
            id,
            status,
            lost_confirmed,
            found_confirmed
        FROM matches
        WHERE lost_item_id = ?
        AND found_item_id = ?
    """, (
        lost[0],
        found[0]
    )).fetchone()

    reasons_text = " | ".join(
        reasons
    )

    if score < 50:
        if existing_match:
            cursor.execute("""
                UPDATE matches
                SET
                    status = 'INACTIVE',
                    score = ?,
                    reasons = ?,
                    name_similarity = ?,
                    description_similarity = ?,
                    image_similarity = ?
                WHERE id = ?
            """, (
                score,
                reasons_text,
                name_pct,
                description_pct,
                image_pct,
                existing_match[0]
            ))

        return

    if existing_match:
        cursor.execute("""
            UPDATE matches
            SET
                score = ?,
                reasons = ?,
                status = 'ACTIVE',
                name_similarity = ?,
                description_similarity = ?,
                image_similarity = ?
            WHERE id = ?
        """, (
            score,
            reasons_text,
            name_pct,
            description_pct,
            image_pct,
            existing_match[0]
        ))

        return

    cursor.execute("""
        INSERT INTO matches
        (
            lost_item_id,
            found_item_id,
            score,
            reasons,
            status,
            name_similarity,
            description_similarity,
            image_similarity
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        lost[0],
        found[0],
        score,
        reasons_text,
        "ACTIVE",
        name_pct,
        description_pct,
        image_pct
    ))

    match_id = cursor.lastrowid

    if lost[9] is not None:
        cursor.execute("""
            INSERT INTO notifications
            (
                user_id,
                match_id,
                message
            )
            VALUES (?, ?, ?)
        """, (
            lost[9],
            match_id,
            "A possible match has been found for your lost item."
        ))

    if found[9] is not None:
        cursor.execute("""
            INSERT INTO notifications
            (
                user_id,
                match_id,
                message
            )
            VALUES (?, ?, ?)
        """, (
            found[9],
            match_id,
            "A possible match has been found for the item you reported."
        ))


def process_new_item(
    item_type,
    new_id
):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    if item_type == "lost":
        new_item = cursor.execute(
            "SELECT * FROM lost_items WHERE id = ?",
            (new_id,)
        ).fetchone()

        opposite_items = cursor.execute(
            "SELECT * FROM found_items WHERE status = 'ACTIVE' AND deleted = 0"
        ).fetchall()

        opposite_type = "found"

    else:
        new_item = cursor.execute(
            "SELECT * FROM found_items WHERE id = ?",
            (new_id,)
        ).fetchone()

        opposite_items = cursor.execute(
            "SELECT * FROM lost_items WHERE status = 'ACTIVE' AND deleted = 0"
        ).fetchall()

        opposite_type = "lost"

    if new_item is None:
        conn.close()
        return

    new_vectors = get_or_compute_embeddings(
        cursor,
        item_type,
        new_item[0],
        new_item[1],
        new_item[2],
        new_item[5]
    )

    for opposite in opposite_items:
        if (
            new_item[9] is not None
            and opposite[9] is not None
            and new_item[9] == opposite[9]
        ):
            continue

        opposite_vectors = get_or_compute_embeddings(
            cursor,
            opposite_type,
            opposite[0],
            opposite[1],
            opposite[2],
            opposite[5]
        )

        if item_type == "lost":
            lost = new_item
            found = opposite
            lost_vectors = new_vectors
            found_vectors = opposite_vectors

        else:
            lost = opposite
            found = new_item
            lost_vectors = opposite_vectors
            found_vectors = new_vectors

        result = compute_match(
            lost,
            found,
            lost_vectors,
            found_vectors
        )

        upsert_match_record(
            cursor,
            lost,
            found,
            result[0],
            result[1],
            result[3],
            result[4],
            result[5]
        )

    conn.commit()
    conn.close()


def recompute_all_matches():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    lost_items_rows = cursor.execute(
        "SELECT * FROM lost_items WHERE status = 'ACTIVE' AND deleted = 0"
    ).fetchall()

    found_items_rows = cursor.execute(
        "SELECT * FROM found_items WHERE status = 'ACTIVE' AND deleted = 0"
    ).fetchall()

    for lost in lost_items_rows:
        lost_vectors = get_or_compute_embeddings(
            cursor,
            "lost",
            lost[0],
            lost[1],
            lost[2],
            lost[5]
        )

        for found in found_items_rows:
            if (
                lost[9] is not None
                and found[9] is not None
                and lost[9] == found[9]
            ):
                continue

            found_vectors = get_or_compute_embeddings(
                cursor,
                "found",
                found[0],
                found[1],
                found[2],
                found[5]
            )

            result = compute_match(
                lost,
                found,
                lost_vectors,
                found_vectors
            )

            upsert_match_record(
                cursor,
                lost,
                found,
                result[0],
                result[1],
                result[3],
                result[4],
                result[5]
            )

    conn.commit()
    conn.close()


@app.route("/recompute-matches")
def recompute_matches_route():
    if "user_id" not in session:
        return redirect("/login")

    try:
        recompute_all_matches()

        flash(
            "AI matching system updated successfully.",
            "success"
        )

    except Exception as error:
        flash(
            f"Matching update failed: {error}",
            "error"
        )

    return redirect("/matches")


@app.route("/matches")
def matches_page():
    if "user_id" not in session:
        return redirect("/login")

    user_id = session["user_id"]

    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            m.id,
            m.score,
            m.reasons,
            m.status,
            m.name_similarity,
            m.description_similarity,
            m.image_similarity,
            l.id,
            l.item_name,
            l.description,
            l.location,
            l.date,
            l.image,
            l.category,
            f.id,
            f.item_name,
            f.description,
            f.location,
            f.date,
            f.image,
            f.category
        FROM matches m
        JOIN lost_items l
            ON m.lost_item_id = l.id
        JOIN found_items f
            ON m.found_item_id = f.id
        WHERE m.status = 'ACTIVE'
        AND l.status = 'ACTIVE'
        AND f.status = 'ACTIVE'
        AND l.deleted = 0
        AND f.deleted = 0
        AND (
            l.user_id = ?
            OR f.user_id = ?
        )
        ORDER BY m.score DESC, l.id DESC
    """, (
        user_id,
        user_id
    ))

    saved_matches = cursor.fetchall()
    conn.close()

    grouped_matches = {}

    for match in saved_matches:
        lost_key = (
            normalize_text(match[8]),
            normalize_text(match[9]),
            normalize_text(match[10]),
            match[11]
        )

        reasons = (
            match[2].split(" | ")
            if match[2]
            else []
        )

        match_data = [
            (
                match[7],
                match[8],
                match[9],
                match[10],
                match[11],
                match[12],
                None,
                None,
                match[13]
            ),
            (
                match[14],
                match[15],
                match[16],
                match[17],
                match[18],
                match[19],
                None,
                None,
                match[20]
            ),
            int(match[1]),
            None,
            reasons,
            match[0],
            match[4] or 0,
            match[5] or 0,
            match[6] or 0
        ]

        if lost_key not in grouped_matches:
            grouped_matches[lost_key] = {
                "lost": match_data[0],
                "matches": []
            }

        grouped_matches[lost_key]["matches"].append(
            match_data
        )

    for group in grouped_matches.values():
        group["matches"].sort(
            key=lambda item: item[2],
            reverse=True
        )

        group["matches"] = group["matches"][:5]

    grouped_matches = list(
        grouped_matches.values()
    )

    return render_template(
        "matches.html",
        grouped_matches=grouped_matches
    )


@app.route("/notifications")
def notifications():
    if "user_id" not in session:
        return redirect("/login")

    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            id,
            match_id,
            message,
            is_read,
            created_at
        FROM notifications
        WHERE user_id = ?
        ORDER BY created_at DESC
    """, (
        session["user_id"],
    ))

    notification_rows = cursor.fetchall()

    cursor.execute("""
        UPDATE notifications
        SET is_read = 1
        WHERE user_id = ?
    """, (
        session["user_id"],
    ))

    conn.commit()
    conn.close()

    return render_template(
        "notifications.html",
        notifications=notification_rows
    )


@app.route(
    "/match/<int:match_id>",
    methods=["GET", "POST"]
)
def match_details(match_id):
    if "user_id" not in session:
        return redirect("/login")

    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            m.id,
            m.score,
            m.reasons,
            m.status,
            m.lost_confirmed,
            m.found_confirmed,
            l.id,
            l.item_name,
            l.description,
            l.location,
            l.date,
            l.image,
            l.latitude,
            l.longitude,
            l.category,
            l.user_id,
            f.id,
            f.item_name,
            f.description,
            f.location,
            f.date,
            f.image,
            f.latitude,
            f.longitude,
            f.category,
            f.user_id,
            m.name_similarity,
            m.description_similarity,
            m.image_similarity
        FROM matches m
        JOIN lost_items l
            ON m.lost_item_id = l.id
        JOIN found_items f
            ON m.found_item_id = f.id
        WHERE m.id = ?
    """, (
        match_id,
    ))

    match = cursor.fetchone()

    if not match:
        conn.close()
        return "Match not found."

    lost_owner_id = match[15]
    found_owner_id = match[25]
    current_user_id = session["user_id"]

    if (
        lost_owner_id != current_user_id
        and found_owner_id != current_user_id
    ):
        conn.close()
        return "You are not authorized to view this match."

    if request.method == "POST":
        if current_user_id == lost_owner_id:
            cursor.execute("""
                UPDATE matches
                SET lost_confirmed = 1
                WHERE id = ?
            """, (
                match_id,
            ))

        elif current_user_id == found_owner_id:
            cursor.execute("""
                UPDATE matches
                SET found_confirmed = 1
                WHERE id = ?
            """, (
                match_id,
            ))

        confirmations = cursor.execute("""
            SELECT
                lost_confirmed,
                found_confirmed
            FROM matches
            WHERE id = ?
        """, (
            match_id,
        )).fetchone()

        if (
            confirmations
            and confirmations[0] == 1
            and confirmations[1] == 1
        ):
            cursor.execute("""
                UPDATE matches
                SET status = 'RESOLVED'
                WHERE id = ?
            """, (
                match_id,
            ))

            cursor.execute("""
                UPDATE lost_items
                SET
                    status = 'RESOLVED',
                    deleted = 1
                WHERE id = ?
            """, (
                match[6],
            ))

            cursor.execute("""
                UPDATE found_items
                SET
                    status = 'RESOLVED',
                    deleted = 1
                WHERE id = ?
            """, (
                match[16],
            ))

            cursor.execute("""
                INSERT INTO notifications
                (
                    user_id,
                    match_id,
                    message
                )
                VALUES (?, ?, ?)
            """, (
                lost_owner_id,
                match_id,
                "The lost and found match has been confirmed and resolved."
            ))

            if found_owner_id != lost_owner_id:
                cursor.execute("""
                    INSERT INTO notifications
                    (
                        user_id,
                        match_id,
                        message
                    )
                    VALUES (?, ?, ?)
                """, (
                    found_owner_id,
                    match_id,
                    "The lost and found match has been confirmed and resolved."
                ))

        conn.commit()
        conn.close()

        return redirect(
            "/match/" + str(match_id)
        )

    reasons = []

    if match[2]:
        reasons = match[2].split(
            " | "
        )

    distance = None

    if (
        match[12] is not None
        and match[13] is not None
        and match[22] is not None
        and match[23] is not None
    ):
        try:
            distance = calculate_distance(
                match[12],
                match[13],
                match[22],
                match[23]
            )

        except (
            ValueError,
            TypeError
        ):
            distance = None

    name_similarity_pct = match[26] or 0
    description_similarity_pct = match[27] or 0
    image_similarity_pct = match[28] or 0

    conn.close()

    return render_template(
        "match_details.html",
        match=match,
        reasons=reasons,
        distance=distance,
        name_similarity=name_similarity_pct,
        description_similarity=description_similarity_pct,
        image_similarity=image_similarity_pct
    )


@app.route(
    "/item/<item_type>/<int:item_id>/close",
    methods=["POST"]
)
def close_item(item_type, item_id):
    if "user_id" not in session:
        return redirect("/login")

    if item_type not in {
        "lost",
        "found"
    }:
        return "Invalid item type."

    table = (
        "lost_items"
        if item_type == "lost"
        else "found_items"
    )

    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    cursor.execute(
        f"""
        UPDATE {table}
        SET status = 'RESOLVED'
        WHERE id = ?
        AND user_id = ?
        AND status = 'ACTIVE'
        AND deleted = 0
        """,
        (
            item_id,
            session["user_id"]
        )
    )

    conn.commit()
    conn.close()

    flash(
        "Report closed successfully.",
        "success"
    )

    return redirect(
        "/my-reports"
    )


@app.route(
    "/item/<item_type>/<int:item_id>/delete",
    methods=["POST"]
)
def delete_item(item_type, item_id):
    if "user_id" not in session:
        return redirect("/login")

    if item_type not in {
        "lost",
        "found"
    }:
        return "Invalid item type."

    table = (
        "lost_items"
        if item_type == "lost"
        else "found_items"
    )

    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    item = cursor.execute(
        f"""
        SELECT
            id,
            user_id
        FROM {table}
        WHERE id = ?
        AND deleted = 0
        """,
        (item_id,)
    ).fetchone()

    if not item:
        conn.close()
        return redirect("/my-reports")

    if item[1] != session["user_id"]:
        conn.close()
        return "You are not authorized to delete this report."

    if item_type == "lost":
        cursor.execute("""
            UPDATE matches
            SET status = 'INACTIVE'
            WHERE lost_item_id = ?
            AND status = 'ACTIVE'
        """, (item_id,))

    else:
        cursor.execute("""
            UPDATE matches
            SET status = 'INACTIVE'
            WHERE found_item_id = ?
            AND status = 'ACTIVE'
        """, (item_id,))

    cursor.execute(
        f"""
        UPDATE {table}
        SET
            status = 'CLOSED',
            deleted = 1
        WHERE id = ?
        AND user_id = ?
        """,
        (
            item_id,
            session["user_id"]
        )
    )

    cursor.execute("""
        DELETE FROM item_embeddings
        WHERE item_type = ?
        AND item_id = ?
    """, (
        item_type,
        item_id
    ))

    conn.commit()
    conn.close()

    flash(
        "Report deleted successfully.",
        "success"
    )

    return redirect(
        "/my-reports"
    )


@app.route("/api/notifications-count")
def notification_count():
    if "user_id" not in session:
        return jsonify({
            "count": 0
        })

    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT COUNT(*)
        FROM notifications
        WHERE user_id = ?
        AND is_read = 0
    """, (
        session["user_id"],
    ))

    count = cursor.fetchone()[0]

    conn.close()

    return jsonify({
        "count": count
    })


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename
    )


if __name__ == "__main__":
    init_db()

    app.run(
        debug=True,
        port=5001
    )