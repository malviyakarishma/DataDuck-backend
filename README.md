# 🦆 DataDuck Backend — Ask. Dig. Discover.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Pydantic](https://img.shields.io/badge/Pydantic-v2-E92063.svg?logo=pydantic&logoColor=white)](https://docs.pydantic.dev/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Security](https://img.shields.io/badge/SQLGlot-Read--Only_AST-brightgreen.svg)](https://github.com/tobymao/sqlglot)

> **DataDuck Backend** is an enterprise-grade FastAPI service that transforms natural language questions into safe, read-only SQL and MongoDB MQL queries, executes them against your connected databases, and streams natural language insights and dynamic chart data using local or cloud AI models.

---

## 🌟 Key Features

- 🧠 **Multi-AI Provider Support**: Seamlessly switch between local offline LLMs (**Ollama**) and cloud providers (**Google Gemini**, **OpenAI**, **Groq**).
- 🗄️ **Multi-Engine Database Connectivity**: Connect and analyze PostgreSQL, MySQL, SQLite, and MongoDB.
- 🛡️ **Zero-Trust Read-Only AST Engine**: Enforces strict AST-level SQL validation using `SQLGlot`. Hard-blocks `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `TRUNCATE`, and multi-statement execution.
- 🔐 **Fernet Encryption at Rest**: Encrypts database connection credentials using AES-128-CBC via Fernet symmetric encryption.
- 📊 **Smart Data & Visualization Pipeline**: Automatically infers dataset characteristics to recommend optimal charts (Line, Bar, Pie, Scatter, Area) and produce executive summary insights.
- 🔑 **JWT Security & Rate Limiting**: Full auth suite (access + refresh tokens, bcrypt hashing) with IP rate limiting via `slowapi`.
- ⚡ **Asynchronous Performance**: Built with async drivers (`asyncpg`, `aiomysql`, `aiosqlite`, `motor`) for high concurrency and fast response times.

---

## 🔌 Supported AI Providers & Databases

### 🤖 AI Models & Engines

| Provider | Supported Models | Recommended For | Privacy |
| :--- | :--- | :--- | :--- |
| **Ollama** *(Local)* | `qwen2.5-coder:7b`, `llama3`, `deepseek-r1` | 100% Offline / Private | 🔒 Local Only (No cloud leakage) |
| **Google Gemini** | `gemini-2.0-flash`, `gemini-1.5-pro` | High Speed & Deep Reasoning | ☁️ Cloud API |
| **OpenAI** | `gpt-4o-mini`, `gpt-4o` | Complex SQL Schema Mapping | ☁️ Cloud API |
| **Groq** | `llama-3.3-70b-versatile`, `mixtral-8x7b-32768` | Ultra Low-Latency Response | ☁️ Cloud API |

### 🗃️ Supported Databases

- **PostgreSQL**: `postgresql+asyncpg://`
- **MySQL**: `mysql+aiomysql://`
- **SQLite**: `sqlite+aiosqlite:///`
- **MongoDB**: `mongodb://` / `mongodb+srv://`

---

## 🏗️ Architecture Overview

```mermaid
flowchart TD
    Client[Frontend Client / User] -->|HTTP / JSON + JWT| API[FastAPI Gateway / App]
    API --> RateLimiter[Slowapi Rate Limiter]
    RateLimiter --> AuthMiddleware[JWT & Fernet Encryption]
    
    AuthMiddleware --> Router{API Router}
    Router -->|/api/auth| AuthSvc[Auth Service]
    Router -->|/api/databases| DBSvc[Database Service]
    Router -->|/api/chat| ChatSvc[Chat & Query Pipeline]
    
    ChatSvc --> SchemaSvc[Schema Inspector & Metadata Summarizer]
    SchemaSvc --> AIBridge[AI Provider Manager]
    
    AIBridge -->|Local AI| Ollama[Ollama Local Service :11434]
    AIBridge -->|Cloud AI| Gemini[Google Gemini API]
    AIBridge -->|Cloud AI| OpenAI[OpenAI API]
    AIBridge -->|Cloud AI| Groq[Groq API]
    
    ChatSvc --> ASTValidator[SQLGlot AST Safe Parser]
    ASTValidator -->|Passed Read-Only Check| Adapter[Database Adapter Factory]
    ASTValidator -->|Destructive Query Detected| Block[400 Security Exception]
    
    Adapter --> TargetDB[(Postgres / MySQL / SQLite / Mongo)]
```

---

## 🚀 Quick Start Guide

### Prerequisites

- **Python 3.10+** installed
- **Git**
- *(Optional)* **Ollama** installed if running local AI models ([ollama.com/download](https://ollama.com/download))

### 1. Clone the Repository

```bash
git clone https://github.com/malviyakarishma/ai-db-backend.git
cd ai-db-backend
```

### 2. Set Up Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Generate a secure encryption key for storing credentials:

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Paste the generated string into `.env` under `ENCRYPTION_KEY`.

#### Example `.env` Configuration (Local Ollama Setup):

```env
DATABASE_URL=postgresql+asyncpg://postgres:password@localhost:5432/dataduck
SECRET_KEY=super-secret-jwt-key-change-in-production
ENCRYPTION_KEY=your_generated_fernet_key_here

# AI Provider Configuration
AI_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5-coder:7b

# Security Settings
MAX_QUERY_ROWS=10000
QUERY_TIMEOUT_SECONDS=30
ALLOWED_ORIGINS=["http://localhost:3000"]
```

### 5. Running Local AI with Ollama (Recommended)

1. Start Ollama service on your machine.
2. Pull the recommended coding model:
   ```bash
   ollama pull qwen2.5-coder:7b
   ```
3. Test Ollama execution:
   ```bash
   ollama run qwen2.5-coder:7b
   ```

### 6. Run the FastAPI Server

```bash
uvicorn app.main:app --reload --port 8000
```

The API server will be available at: `http://localhost:8000`

---

## 🔒 Security & Read-Only Guarantee

DataDuck prioritizes database safety above all else. Every generated query undergoes multi-layered inspection before execution:

1. **AST-Level Verification (`SQLGlot`)**:
   - Only `SELECT` statements and `WITH ... SELECT` Common Table Expressions (CTEs) are allowed.
   - Any query containing DDL/DML keywords (`INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `TRUNCATE`, `CREATE`, `GRANT`, `REVOKE`, `COPY`, `CALL`, `EXECUTE`) is immediately rejected.
2. **Multi-Statement Blocking**: Multiple statements separated by semicolons are automatically disallowed.
3. **Execution Sandboxing**: All user queries run under strict query execution timeouts (`QUERY_TIMEOUT_SECONDS=30`) and output caps (`MAX_QUERY_ROWS=10000`).
4. **Credential Isolation**: Database connection strings and credentials are encrypted symmetrically at rest using Fernet keys.

---

## 📡 API Reference & Endpoints

Interactive Swagger UI documentation is available at `http://localhost:8000/api/docs` (when `DEBUG=true`).

### 🟢 Health Checks
- `GET /api/health` — Base backend health status.
- `GET /api/health/ollama` — Verifies local Ollama service connection and model availability.

### 🔑 Authentication (`/api/auth`)
- `POST /api/auth/register` — Register a new user account.
- `POST /api/auth/login` — Authenticate and receive JWT access token.
- `POST /api/auth/refresh` — Refresh access token.
- `GET /api/auth/me` — Retrieve current user profile.

### 🗄️ Database Management (`/api/databases`)
- `POST /api/databases/test` — Test connection parameters without saving.
- `POST /api/databases` — Save a database connection (encrypted).
- `GET /api/databases` — List saved connections for current user.
- `GET /api/databases/{id}/schema` — Retrieve full schema metadata (tables, columns, types, foreign keys).

### 💬 Chat & Query Engine (`/api/chat`)
- `POST /api/chat/conversations` — Start a new chat conversation.
- `GET /api/chat/conversations` — Fetch chat history threads.
- `POST /api/chat/conversations/{id}/messages` — Send natural language query; generates query, executes safely, generates chart configuration, and streams insights.

---

## 📁 Repository Structure

```
ai-db-backend/
├── app/
│   ├── api/                  # API routers (auth, databases, chat)
│   ├── core/                 # Core config, DB session, exception handlers
│   ├── database_adapters/   # Adapters for Postgres, MySQL, SQLite, MongoDB
│   ├── models/               # SQLAlchemy ORM models (User, Connection, Chat)
│   ├── schemas/              # Pydantic schemas for request/response validation
│   ├── security/             # JWT auth, Fernet encryption, SQLGlot query validator
│   ├── services/             # Business logic (AI providers, query execution, schema engine)
│   └── main.py               # FastAPI entry point & CORS configuration
├── .env.example              # Template environment file
├── requirements.txt          # Python dependencies
└── README.md                 # Documentation
```

---

## 🧪 Testing Prompts

### Safe Read-Only Queries:
- `"How many total users signed up last month?"`
- `"What are the top 5 highest grossing products?"`
- `"Show monthly revenue trends for 2024 as a line chart."`

### Blocked Security Prompts (Verification):
- `"Delete all inactive accounts."`
- `"Drop the transactions table."`
- `"Update user balances set balance = 0."`

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:
1. Fork the project.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

