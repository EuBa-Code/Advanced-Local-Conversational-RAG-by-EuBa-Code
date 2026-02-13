"""
Test dataset for RAGAS evaluation of the RAG system.
Each entry contains a question, the expected answer (ground truth),
and the source file containing the information.
"""

EVALUATION_DATASET = [
    # --- Corporate Governance Questions (doc 01) ---
    {
        "question": "Who founded Aetheria Global Solutions and where?",
        "ground_truth": (
            "Aetheria Global Solutions was founded in 2014 in Zurich "
            "by Dr. Aris Thorne and lead systems engineer Sarah Jenkins."
        ),
        "source_file": "01_corporate_governance_vision.txt",
    },
    {
        "question": "What is the organizational structure model used by AGS?",
        "ground_truth": (
            "AGS operates on a 'Holocratic Matrix' organized into autonomous "
            "'Impact Circles' rather than traditional top-down departments."
        ),
        "source_file": "01_corporate_governance_vision.txt",
    },

    # --- Technical Infrastructure Questions (doc 02) ---
    {
        "question": "What programming language is the Nexus engine written in and why?",
        "ground_truth": (
            "The Nexus engine is written entirely in Rust to ensure memory safety, "
            "high concurrency, and performance close to the metal."
        ),
        "source_file": "02_nexus_technical_infrastructure.txt",
    },
    {
        "question": "What vector database does AGS use and what indexing method?",
        "ground_truth": (
            "AGS uses Qdrant running in distributed mode with HNSW "
            "(Hierarchical Navigable Small World) indexing for the semantic layer."
        ),
        "source_file": "02_nexus_technical_infrastructure.txt",
    },

    # --- HR & Culture Questions (doc 03) ---
    {
        "question": "What is the paternity leave policy at AGS?",
        "ground_truth": (
            "AGS offers paternity leave as part of their benefits package. "
            "The specific details are outlined in the HR benefits documentation."
        ),
        "source_file": "03_hr_benefits_culture.txt",
    },

    # --- Cybersecurity Questions (doc 05) ---
    {
        "question": "What authentication method does AGS use for production environments?",
        "ground_truth": (
            "AGS requires hardware security keys (FIDO2/YubiKey) for access "
            "to sensitive production environments. Password-based authentication "
            "has been eliminated for all critical systems."
        ),
        "source_file": "05_cybersecurity_zero_trust.txt",
    },
    {
        "question": "What is AGS's Recovery Time Objective (RTO)?",
        "ground_truth": (
            "AGS's Recovery Time Objective (RTO) is 45 minutes, with a Recovery "
            "Point Objective (RPO) of less than 5 seconds, achieved through "
            "real-time synchronous transaction log replication."
        ),
        "source_file": "05_cybersecurity_zero_trust.txt",
    },

    # --- Financial & ESG Questions (doc 06) ---
    {
        "question": "What is AGS's approach to ESG and sustainability?",
        "ground_truth": (
            "AGS is committed to reducing the energy cost per computation cycle "
            "by 15% annually through hardware-software co-design as part of their "
            "Thermodynamic Efficiency core value."
        ),
        "source_file": "06_financial_sustainability_esg.txt",
    },

    # --- Customer Support SLA Questions (doc 10) ---
    {
        "question": "What is the response time for a P0 critical incident at AGS?",
        "ground_truth": (
            "For P0 (Critical) incidents such as total outage or confirmed data breach, "
            "the response time is 15 minutes with a resolution target of less than 4 hours."
        ),
        "source_file": "10_customer_support_sla.txt",
    },
    {
        "question": "What uptime does AGS guarantee for Enterprise clients?",
        "ground_truth": (
            "AGS guarantees a monthly uptime of 99.99% for Enterprise and Sovereign clients. "
            "If not met, service credits are provided: 10% for 99.9-99.99%, "
            "25% for 99.0-99.9%, and 50% for below 99.0%."
        ),
        "source_file": "10_customer_support_sla.txt",
    },
]
