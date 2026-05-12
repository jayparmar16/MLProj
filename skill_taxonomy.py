"""Hand-curated Skill / Knowledge taxonomy used as an inference-time post-processor.

Motivation: SkillSpan's gold annotations are inconsistent on the Skill vs.
Knowledge split for named technologies (Python is B-Skill in some examples and
B-Knowledge in others, depending on whether the sentence frames it as an
ability or as a body of knowledge). The student learned that ambiguity and now
defaults to Knowledge for almost every named programming language, cloud
provider, or framework. This file gives `infer.py` an authoritative override:
if the extracted span maps cleanly to an entry in SKILL_TERMS, force the
category to Skill. Same logic for KNOWLEDGE_TERMS in the opposite direction.

This is a stand-in for O*NET's Technology Skills / Knowledge taxonomy. A
production version should pull from the official O*NET bulk download
(www.onetcenter.org/database.html). This list is intentionally conservative:
only well-known, unambiguous items are included.

Matching is case-insensitive, lemma-tolerant for common plurals, and considers
common aliases (k8s -> Kubernetes, JS -> JavaScript, etc.).
"""

# ---------------------------------------------------------------------------
# Skill: programming languages, frameworks, cloud providers, devops tools,
# databases, version control, OSes, build/test tooling. Per ESCO / SkillSpan,
# "Skill" can also include soft competences (communication, leadership), but
# the taxonomy below is focused on named technologies where the model most
# often disagrees with the human-intuitive category.
# ---------------------------------------------------------------------------
SKILL_TERMS = {
    # Programming languages
    "python", "java", "javascript", "typescript", "c", "c++", "c#", "go", "golang",
    "rust", "ruby", "php", "swift", "kotlin", "scala", "r", "matlab", "sql",
    "bash", "shell", "powershell", "perl", "lua", "dart", "elixir", "haskell",
    "clojure", "groovy", "objective-c", "f#", "ocaml", "julia", "cobol",
    "fortran", "assembly", "html", "css", "sass", "scss", "less",
    # Web frameworks
    "react", "vue", "vue.js", "angular", "svelte", "next.js", "nextjs",
    "nuxt.js", "nuxtjs", "express", "express.js", "django", "flask", "fastapi",
    "spring", "spring boot", "asp.net", "rails", "ruby on rails", "laravel",
    "symfony", "jquery", "ember", "backbone",
    # Mobile
    "react native", "flutter", "swiftui", "xamarin", "ionic", "android",
    "ios",
    # ML / data frameworks
    "pytorch", "tensorflow", "keras", "scikit-learn", "sklearn", "pandas",
    "numpy", "scipy", "jax", "xgboost", "lightgbm", "catboost", "mlflow",
    "kubeflow", "huggingface", "transformers", "spacy", "nltk", "opencv",
    # Cloud providers + major services
    "aws", "amazon web services", "azure", "microsoft azure", "gcp",
    "google cloud", "google cloud platform", "ibm cloud", "oracle cloud",
    "digitalocean", "heroku", "vercel", "netlify", "cloudflare", "linode",
    "ec2", "s3", "lambda", "sagemaker", "rds", "dynamodb", "eks", "ecs",
    "fargate", "glue", "redshift", "athena", "kinesis", "cloudfront",
    "cloudwatch", "iam", "vpc", "route53",
    # Containers / orchestration
    "docker", "kubernetes", "k8s", "openshift", "helm", "podman", "containerd",
    "rancher", "nomad",
    # CI / CD
    "jenkins", "github actions", "gitlab ci", "circleci", "travis ci",
    "travisci", "azure devops", "argo", "argocd", "tekton", "spinnaker",
    "drone",
    # Infrastructure as code
    "terraform", "ansible", "puppet", "chef", "cloudformation", "pulumi",
    "salt", "saltstack",
    # Monitoring / observability tools
    "prometheus", "grafana", "datadog", "new relic", "splunk", "elk",
    "elasticsearch", "logstash", "kibana", "jaeger", "zipkin", "opentelemetry",
    "pagerduty", "victoriametrics",
    # Databases
    "postgresql", "postgres", "mysql", "mariadb", "mongodb", "redis",
    "cassandra", "couchdb", "neo4j", "bigquery", "snowflake", "databricks",
    "oracle db", "oracle database", "sql server", "mssql", "sqlite", "hbase",
    "clickhouse", "influxdb", "timescaledb",
    # Big data / streaming
    "spark", "apache spark", "hadoop", "hdfs", "kafka", "apache kafka",
    "flink", "apache flink", "airflow", "apache airflow", "beam",
    "apache beam", "storm", "presto", "trino", "dbt",
    # Version control
    "git", "github", "gitlab", "bitbucket", "subversion", "svn", "mercurial",
    # OS
    "linux", "unix", "windows", "macos", "ubuntu", "centos", "debian",
    "rhel", "red hat enterprise linux", "alpine", "fedora", "freebsd",
    # Build / package managers
    "webpack", "vite", "rollup", "parcel", "esbuild", "maven", "gradle",
    "npm", "yarn", "pnpm", "bazel", "make", "cmake",
    # Testing
    "jest", "mocha", "chai", "jasmine", "pytest", "junit", "testng",
    "selenium", "cypress", "playwright", "puppeteer", "rspec", "karma",
    # APIs / protocols
    "graphql", "rest", "grpc", "websocket", "oauth", "jwt", "openapi",
    "swagger", "soap",
    # Misc tools
    "tableau", "power bi", "powerbi", "looker", "metabase", "superset",
    "jira", "confluence", "slack", "notion", "figma", "sketch", "adobe xd",
    "postman", "insomnia",
}

# Common aliases / abbreviations that should map to canonical Tech terms.
SKILL_ALIASES = {
    "k8s": "kubernetes",
    "js": "javascript",
    "ts": "typescript",
    "py": "python",
    "pg": "postgresql",
    "tf": "tensorflow",
    "rn": "react native",
}

# ---------------------------------------------------------------------------
# Knowledge: academic domains, methodologies, soft skills, conceptual fields.
# These should always be Knowledge, never Tech, regardless of model prediction.
# ---------------------------------------------------------------------------
KNOWLEDGE_TERMS = {
    # Mathematics / stats
    "statistics", "linear algebra", "calculus", "probability", "mathematics",
    "discrete mathematics", "optimization", "numerical analysis",
    "differential equations", "geometry",
    # CS / ML concepts (the fields, not the tools)
    "machine learning", "deep learning", "reinforcement learning",
    "natural language processing", "nlp", "computer vision", "data science",
    "data engineering", "data analytics", "data analysis",
    "algorithms", "data structures", "distributed systems", "operating systems",
    "computer networks", "cryptography", "computer architecture",
    "software engineering", "software architecture", "system design",
    "cloud architecture", "security", "cybersecurity", "information security",
    # Methodologies / processes
    "agile", "scrum", "kanban", "waterfall", "lean", "six sigma",
    "test-driven development", "tdd", "behaviour-driven development", "bdd",
    "domain-driven design", "ddd", "microservices", "devops", "mlops",
    "project management", "product management", "stakeholder management",
    "risk management", "change management",
    # Business / domain knowledge
    "finance", "accounting", "marketing", "sales", "operations", "logistics",
    "supply chain", "consumer psychology", "market research", "branding",
    "brand strategy", "campaign management",
    # Soft skills (frequently mislabelled as Tech)
    "communication", "verbal communication", "written communication",
    "leadership", "teamwork", "collaboration", "problem solving",
    "problem-solving", "critical thinking", "analytical thinking",
    "presentation", "negotiation", "mentoring", "coaching",
    "time management", "decision making", "decision-making",
    "interpersonal skills", "emotional intelligence", "adaptability",
    "creativity", "attention to detail",
    # Languages (the linguistic kind, not programming)
    "english", "spanish", "french", "german", "mandarin", "japanese",
    "portuguese", "italian", "arabic", "hindi", "russian",
}


def _normalize(text):
    """Lower, strip, collapse internal whitespace, strip trailing 's' for naive plural handling."""
    t = " ".join(text.lower().split())
    if t.endswith("s") and len(t) > 3 and not t.endswith("ss"):
        # Naive de-pluralization — only strips a trailing 's' that's not part of 'ss'
        t_singular = t[:-1]
    else:
        t_singular = t
    return t, t_singular


def classify_span(span_text):
    """Return 'Skill', 'Knowledge', or None if no confident match.

    Used by infer.py to override the model's category prediction when the
    extracted phrase is unambiguous under the taxonomy.
    """
    if not span_text:
        return None
    norm, singular = _normalize(span_text)

    # Alias resolution first
    aliased = SKILL_ALIASES.get(norm) or SKILL_ALIASES.get(singular)
    if aliased:
        return "Skill"

    if norm in SKILL_TERMS or singular in SKILL_TERMS:
        return "Skill"
    if norm in KNOWLEDGE_TERMS or singular in KNOWLEDGE_TERMS:
        return "Knowledge"
    return None
