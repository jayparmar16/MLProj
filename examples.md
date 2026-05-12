# Inference Examples

All outputs below come from a single trained checkpoint: `./checkpoints/best_model.pt` (test macro F1 ≈ 0.63). 

Each example is annotated with a one-line **commentary** describing the notable hits and misses.

---

## 1. Senior Backend Engineer (in-domain — software)

**JD**

> We are looking for a Senior Backend Engineer to join our core infrastructure team. You must have strong problem-solving skills and excellent communication abilities to work with cross-functional teams. Experience with Python, Django, and PostgreSQL is required. Familiarity with containerization technologies like Docker and Kubernetes is a huge plus. You should also be capable of mentoring junior developers and leading architectural discussions.

**Skill**
- infrastructure
- problem
- solving skills
- communication abilities
- work with cross
- Python
- Django
- Docker
- Kubernetes
- mentoring junior developers
- leading architectural discussions

**Knowledge**
- containerization

**Commentary.** Strongest in-domain example. All four named technologies (Python, Django, Docker, Kubernetes) land correctly in Skill. Soft skills (communication, mentoring, leading) also correctly in Skill. Span boundaries are rough: "problem" / "solving skills" should be one span, and "work with cross" is cut off. PostgreSQL is missed entirely.

---

## 2. ML / Data Science Engineer

**JD**

> We are hiring a Machine Learning Engineer with deep experience in PyTorch, TensorFlow, and scikit-learn. You will build and deploy NLP models in production. Strong knowledge of statistics, linear algebra, and probability is required. Familiarity with AWS SageMaker, MLflow, and distributed training is a plus.

**Skill**
- TensorFlow
- learn
- build and deploy NLP models in production
- AWS
- MLflow

**Knowledge**
- scikit
- statistics
- linear
- algebra
- probability
- distributed

**Commentary.** TensorFlow, AWS, MLflow correctly in Skill (taxonomy override). Academic terms (statistics, linear algebra, probability) correctly Knowledge. "scikit-learn" got split: "scikit" lands in Knowledge and "learn" lands in Skill — a boundary error. PyTorch missed entirely. SageMaker missed.

---

## 3. Senior Frontend Developer

**JD**

> Looking for a Senior Frontend Engineer fluent in React, TypeScript, and modern CSS. You should have experience with Redux, Next.js, and unit testing using Jest. Knowledge of accessibility standards and responsive design is essential. Familiarity with GraphQL and Webpack is a bonus.

**Skill**
- TypeScript
- CSS
- Jest
- GraphQL
- Webpack

**Knowledge**
- modern
- Redux
- unit testing
- accessibility standards

**Commentary.** TypeScript, CSS, Jest, GraphQL, Webpack all correctly in Skill via the taxonomy. React and Next.js missed entirely — the model never predicted a span for them. Redux is in the taxonomy file but didn't override here (alias-matching gap; would be fixed by adding "redux" to `SKILL_TERMS`).

---

## 4. Senior DevOps Engineer

**JD**

> Senior DevOps Engineer with extensive experience in Kubernetes, Docker, Terraform, and AWS. You will own CI/CD pipelines using Jenkins and GitHub Actions. Strong scripting skills in Bash and Python required. Knowledge of monitoring stacks (Prometheus, Grafana) and observability is critical.

**Skill**
- Kubernetes
- Docker
- Terraform
- AWS
- Jenkins
- GitHub
- Bash
- Python
- monitoring
- Prometheus
- Grafana

**Knowledge**
- CI
- observability

**Commentary.** Cleanest result of the set: 11 distinct technologies correctly in Skill. "CI" landed in Knowledge (taxonomy doesn't include the abbreviation by itself). "observability" sits in Knowledge — debatable, since it's a concept-plus-tooling area, but the taxonomy treats concepts as Knowledge.

---

## 5. Data Engineer

**JD**

> Hiring a Data Engineer to design and maintain our data lake on Snowflake. You will build ETL pipelines using Apache Airflow, dbt, and Spark. Strong SQL skills are mandatory. Experience with Kafka streaming and AWS Glue is a plus. Knowledge of dimensional modeling and data warehousing best practices is essential.

**Skill**
- Snowflake
- build ETL pipelines
- Spark
- SQL
- Kafka
- AWS

**Knowledge**
- Apache
- streaming
- dimensional modeling
- data warehousing best practices

**Commentary.** Mixed: named tools (Snowflake, Spark, SQL, Kafka, AWS) correctly Skill. Conceptual/methodological terms (dimensional modeling, data warehousing best practices) correctly Knowledge. "Apache Airflow" got split — "Apache" lands in Knowledge alone — a boundary error. dbt and AWS Glue missed entirely.

---

## 6. UX / Product Designer (creative, less software-focused)

**JD**

> Senior Product Designer needed for our SaaS platform. You will own user research, wireframing in Figma, and prototyping. Strong understanding of design systems, typography, and accessibility (WCAG) is required. Experience collaborating with frontend engineers and product managers on agile teams. Background in interaction design and usability testing.

**Skill**
- own
- Figma
- collaborating

**Knowledge**
- user research
- wireframing
- prototyping
- design systems
- typography
- frontend engineers and product managers on
- agile teams
- testing

**Commentary.** Domain stretch: design isn't well-represented in SkillSpan train. Figma correctly Skill (taxonomy). Methodology/concept terms (user research, design systems, typography, agile) reasonably in Knowledge. "frontend engineers and product managers on" is a single weird over-extended span. "own" and "testing" are odd minimal extractions.

---

## 7. Healthcare Data Analyst (out-of-domain — non-software)

**JD**

> We seek a Healthcare Data Analyst to support our clinical operations team. The role requires expertise in epidemiology, biostatistics, and clinical trial design. You will use SAS, R, and SQL to analyze patient outcomes data. Familiarity with HIPAA compliance and electronic health records (EHR) systems is essential. Strong written communication for regulatory submissions required.

**Skill**
- design
- R
- SQL
- analyze
- communication for regulatory submissions

**Knowledge**
- epidemiology
- biostatistics
- SAS
- electronic health records

**Commentary.** Strong on academic / methodological terms (epidemiology, biostatistics, electronic health records correctly Knowledge). R and SQL correctly Skill. SAS landed in Knowledge — taxonomy doesn't cover statistical software, and the model's default is Knowledge for named tools without an override. HIPAA missed.

---

## 8. Marketing Manager (non-tech sanity check)

**JD**

> We are hiring a Marketing Manager to lead our brand strategy and campaign execution. The ideal candidate has strong leadership skills, excellent written and verbal communication, and experience managing cross-functional teams. Knowledge of consumer psychology and market research methodologies is preferred. Comfortable presenting to executives.

**Skill**
- campaign execution
- leadership skills
- managing cross
- presenting

**Knowledge**
- brand strategy
- communication
- consumer psychology
- market research methodologies

**Commentary.** Soft-skill JD — no programming terms at all. The model still fires confidently and the categorisation is reasonable. Soft competences (leadership skills, managing cross, presenting) in Skill, conceptual/strategic terms in Knowledge. "communication" flipped to Knowledge by the taxonomy. The model generalises out of the tech domain better than expected.

---

## 9. Very short JD (boundary case)

**JD**

> Looking for a Java developer with experience in Spring Boot and Maven.

**Skill**
- Spring Boot

**Knowledge**
- *(empty)*

**Commentary.** The model only catches Spring Boot. Java and Maven are missed, even though both are in the taxonomy. The taxonomy can only fix the *category* of a span the model already extracted — it cannot create new spans for terms the BIO output never marked. This is the post-processor's hard limit.

---

## Patterns across the nine examples

| Pattern | Where it shows up |
|---|---|
| **Correctly extracted + correctly categorised** | Docker, Kubernetes, Python, Django, TypeScript, GraphQL, Snowflake, Spark, SQL, Figma, R, epidemiology, biostatistics, brand strategy, consumer psychology |
| **Correctly extracted, taxonomy fixed category** | TensorFlow, AWS, MLflow, CSS, Jest, Webpack, Jenkins, Prometheus, Grafana (all flipped Skill → Knowledge → Skill by the post-processor) |
| **Missed entirely** | PyTorch, React, Next.js, PostgreSQL, dbt, AWS Glue, HIPAA, Java, Maven, SageMaker |
| **Boundary errors** | "scikit" / "learn" (scikit-learn split), "Apache" / "Airflow" split, "problem" / "solving skills" split, "work with cross" cut off |
| **Weird over-extensions** | "frontend engineers and product managers on" (one span), "communication for regulatory submissions" |

**Headline**: the model is reliable when a named term is unambiguous *and* in the model's vocabulary of seen examples. It misses ~30–40% of the technologies named in a typical JD, and its span boundaries on multi-word phrases are inconsistent.
