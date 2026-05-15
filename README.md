<p align="center">
  <img src="frontend/assets/images/icon.png" width="250" alt="SpotGuide logo" />
</p>

# SpotGuide

An AI-powered mobile app for discovering the best places in Warsaw. Ask in natural language and get personalized recommendations for restaurants, cafes, culture, and nightlife.

---

## Screenshots

| AI Chat | Interactive Map | Map with place cards  |
|:---:|:---:|:---:|
| <img src="screenshots/chat.PNG" width="100%" alt="AI Chat" /> | <img src="screenshots/map.PNG" width="100%" alt="Interactive Map" /> | <img src="screenshots/cards.PNG" width="100%" alt="Place Cards" /> |

| Favourites | Place Info | Login |
|:---:|:---:|:---:|
| <img src="screenshots/favourites.PNG" width="100%" alt="Favourites" /> | <img src="screenshots/info.PNG" width="100%" alt="Place Info" /> | <img src="screenshots/login.PNG" width="100%" alt="Login" /> |

---

## Features

- **AI Chat** — ask anything in natural language, get curated place recommendations
- **Interactive Map** — explore venues by category with smart clustering
- **Favourites** — save places and filter by category, price, and district
- **Semantic Search** — finds places based on vibe, not just keywords
- **Secure Auth** — JWT-based authentication via AWS Cognito

---

## Tech Stack

| Layer | Technology |
|---|---|
| Mobile | React Native, Expo, TypeScript |
| Backend | FastAPI, Python |
| AI | OpenAI GPT-4o, instructor, RAG pipeline |
| Auth | AWS Cognito |
| Database | PostgreSQL, TimescaleDB, pgvector |
| Infrastructure | AWS ECS Fargate, Terraform |
| CI/CD | GitHub Actions, OIDC |

---

## How It Works

The app uses a **RAG (Retrieval-Augmented Generation)** pipeline:

1. User sends a message in natural language
2. Message is **classified** — new query, follow-up, or hybrid
3. **Metadata is extracted** — district, price range, opening hours, category
4. Query is **expanded** using HyDE (Hypothetical Document Embeddings)
5. **Vector search** finds matching venues with metadata filters
6. LLM **synthesizes** results into a conversational response

---

## Project Structure

```
├── frontend/
│   ├── app/            # Expo Router screens
│   ├── components/     # UI components
│   ├── api/            # API client (Axios)
│   └── store/          # Zustand state management
│
└── backend/
    ├── routers/        # API endpoints
    ├── services/       # Chat service, RAG pipeline, LLM
    ├── database/       # Repositories, vector store
    └── schemas/        # Pydantic models
```

---

## Getting Started

### Requirements

- Node.js 18+
- Python 3.11+
- PostgreSQL with pgvector
- AWS account (Cognito, ECS, RDS)
- OpenAI API key
- Terraform 1.5+

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in your values
uvicorn main:app --reload
```

### Mobile

```bash
cd frontend
npm install
cp .env.example .env   # fill in your values
npx expo start
```

---

## Environment Variables

```bash
# backend/.env
COGNITO_REGION=eu-north-1
COGNITO_USER_POOL_ID=
COGNITO_APP_CLIENT_ID=
DATABASE_URL=postgresql://user:password@localhost:5432/spotguide
OPENAI_API_KEY=
ENVIRONMENT=dev
```

---

## Infrastructure

Deployed on AWS using Terraform. All infrastructure is defined as code in the `terraform/` directory.

| Service | Role |
|---|---|
| **ECS Fargate** | Containerized backend — no server management |
| **RDS PostgreSQL** | Managed database with pgvector extension |
| **ECR** | Docker image registry |
| **ALB** | HTTPS load balancer with SSL termination via ACM |
| **Route 53** | DNS management |
| **ACM** | TLS certificate (CA-signed, auto-renewed) |
| **Cognito** | User pool, JWT issuance, JWKS key rotation |
| **CloudWatch** | Logs, metrics, and alarms for ECS tasks |
| **IAM** | Least-privilege roles for ECS task and GitHub Actions |

### Provisioning from scratch

```bash
cd terraform
terraform init
terraform plan
terraform apply
```

> HTTP traffic is automatically redirected to HTTPS by the ALB listener. The `ENVIRONMENT=prod` variable disables Swagger UI (`/docs`) on the production endpoint.

---

## CI/CD

Deployments are automated via **GitHub Actions** on every push to `main`.

```
push to main
    └── build Docker image
    └── push to ECR (tagged with git SHA)
    └── update ECS service (rolling deploy)
    └── wait for ECS stabilization
```

Authentication to AWS uses **OIDC** — no long-lived secrets stored in GitHub. The pipeline assumes an IAM role scoped to the specific repository and branch.

Docker layer caching is handled via ECR to keep build times short. To roll back, redeploy the previous image tag:

```bash
aws ecs update-service \
  --cluster spotguide \
  --service backend \
  --force-new-deployment \
  --task-definition spotguide-backend:<previous-revision>
```

---

## Security

- **RS256 JWT validation** with JWKS key rotation via Cognito
- **Per-user data isolation** enforced on all endpoints — `user_id` sourced from verified token only
- **Input validation** via Pydantic on all requests; parameterized SQL queries via psycopg2
- **Security headers** — `X-Content-Type-Options`, `X-Frame-Options`, `HSTS` (max-age=31536000)
- **CORS** — no wildcard origins in production
- **Swagger UI disabled** in production via `ENVIRONMENT=prod`
- Tested against **OWASP Web Top 10:2025**, **API Top 10:2023**, and **Mobile Top 10:2024** — 138 tests passed

---

## Documentation

For detailed technical documentation — architecture, API reference, database schema, infrastructure setup, CI/CD, and security audit — see [`dokumentacja.pdf`](./dokumentacja.pdf).

---

## License

MIT
