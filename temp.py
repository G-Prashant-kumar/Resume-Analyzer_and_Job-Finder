import http.client
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def get_jobs_and_skills(city, role):
    # Connect to the API
    conn = http.client.HTTPSConnection("jobs-api14.p.rapidapi.com")

    headers = {
        'x-rapidapi-key': "212e03f368msh5f7f6cf3c2899fdp106a20jsneb520747745a",
        'x-rapidapi-host': "jobs-api14.p.rapidapi.com"
    }

    # Encode parameters properly
    city_encoded = city.strip().replace(" ", "%20")
    role_encoded = role.strip().replace(" ", "%20")

    # Avoid adding `location=` if city is empty
    if city_encoded:
        url = f"/v2/list?query={role_encoded}&location={city_encoded}&autoTranslateLocation=false&remoteOnly=false&employmentTypes=fulltime%3Bparttime%3Bintern%3Bcontractor"
    else:
        url = f"/v2/list?query={role_encoded}&autoTranslateLocation=false&remoteOnly=false&employmentTypes=fulltime%3Bparttime%3Bintern%3Bcontractor"

    # Request for job data
    conn.request("GET", url, headers=headers)
    res = conn.getresponse()
    data = res.read()

    # Parse the JSON response
    response = json.loads(data.decode("utf-8"))

    # Expanded list of common tech skills and technologies
    skills_keywords = [
    # Technical Skills
    "Java", "JavaScript", "React", "Node.js", "HTML", "CSS", "Python", "SQL", "AWS", "Git", "TypeScript", "Angular",
    "Docker", "Kubernetes", "Ruby", "Swift", "PHP", "Scala", "Go", "Linux", "C++", "C#", "Ruby on Rails", "Vue.js",
    "Django", "Flask", "MongoDB", "MySQL", "PostgreSQL", "SQLite", "Redis", "Terraform", "Azure", "GCP", "JavaFX",
    "Jenkins", "Spring", "Hibernate", "Android", "iOS", "GraphQL", "REST", "Microservices", "Serverless",
    "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch", "Keras", "OpenCV", "Pandas", "NumPy", "Scikit-learn",
    "AWS Lambda", "Apache Kafka", "RabbitMQ", "Elasticsearch", "Oracle", "GitHub", "GitLab", "Jira", "Agile", "Scrum",
    "CI/CD", "DevOps", "Software Development", "API", "Cloud", "NoSQL", "TDD", "BDD", "JUnit", "Postman",
    "AWS S3", "AWS EC2", "AWS RDS", "Docker Compose", "Jupyter", "Tableau", "Power BI", "Kibana", "Fluentd", "Prometheus",
    "MATLAB", "R", "Kotlin", "Shell Scripting", "Perl", "SASS", "LESS", "Bootstrap", "Tailwind CSS", "Express.js",
    "Laravel", "CodeIgniter", "NestJS", "ASP.NET", "Firewalls", "VPNs", "AWS IAM", "OAuth", "SSL/TLS", "Penetration Testing",
    "Wireshark", "Vulnerability Scanning", "Cybersecurity", "Bitbucket", "Asana", "Confluence", "Eclipse", "PyCharm",
    "Hadoop", "Apache Spark", "Snowflake", "Airflow", "Talend", "ETL Pipelines", "Data Warehousing", "Data Analytics",
    "Data Visualization", "Bloomberg Terminal", "QuickBooks", "SAP", "Financial Modeling", "Telemedicine", "HIPAA Compliance",
    "Contract Drafting", "Legal Research", "Automation", "Integration", "Cloud Migration", "Continuous Learning",
    "Scalability", "Innovation", "Stakeholder Management", "Product Development", "Debugging Issues", "Efficient UI Development",
    "Designing Scalable Systems", "Code Optimization", "System Architecture", "User-Centric Design", "Prototype Development",
    "Front-End Development", "Back-End Development", "Full-Stack Development", "Database Design", "Data Cleaning",
    "Feature Engineering", "Distributed Systems", "Performance Tuning", "Testing and Debugging", "Secure Coding",
    "Configuration Management", "Site Reliability Engineering", "Network Security", "Automation Testing",
    "Load Testing", "Regression Testing", "Cross-Browser Testing", "Mobile App Development", "Game Development",
    "AI Algorithms", "Big Data", "Cloud Computing", "Data Mining", "Data Modeling", "Web Development", 
    "Blockchain", "Cryptography", "System Integration", "IoT Development", "Embedded Systems", "Scripting Languages",

    # Soft Skills
    "Communication", "Leadership", "Teamwork", "Problem-Solving", "Critical Thinking", "Adaptability", "Creativity",
    "Time Management", "Conflict Resolution", "Negotiation", "Attention to Detail", "Emotional Intelligence",
    "Project Management", "Interpersonal Skills", "Public Speaking", "Decision-Making", "Organizational Skills",
    "Presentation Skills", "Active Listening", "Multitasking", "Resilience", "Collaboration", "Continual Learning",
    "Consistency", "Dedication", "Strategic Planning", "Mentorship", "Empathy", "Flexibility", "Initiative",
    "Visionary Thinking", "Work Ethic", "Self-Motivation", "Accountability", "Process Improvement",
    "Client Interaction", "Stakeholder Collaboration", "Prioritization", "Stress Management", "Analytical Thinking",
    "Constructive Feedback", "Proactive Problem Solving", "Cultural Awareness", "Team Building", "Training and Development",
    "Customer Service Skills", "Writing Skills", "Facilitation", "Logical Reasoning", "Goal-Oriented",
    "Personal Development", "Networking", "Social Intelligence", "Delegation", "Cross-Functional Collaboration",
    "Conflict Management", "Stress Tolerance"
    ]


    # Function to extract skills from the job description using regex
    def extract_skills_from_description(description):
        found_skills = []
        # Check for skills in the job description
        for skill in skills_keywords:
            if re.search(r'\b' + re.escape(skill) + r'\b', description, re.IGNORECASE):
                found_skills.append(skill)
        return found_skills

    # Prepare the job details with extracted skills
    jobs_data = []
    if "jobs" in response and len(response["jobs"]) > 0:
        for job in response["jobs"]:
            job_title = job.get("title", "N/A")
            company = job.get("company", "N/A")
            location = job.get("location", "N/A")
            employment_type = job.get("employmentType", "N/A")
            description = job.get("description", "")
            job_url = job.get("jobProviders", [{}])[0].get("url", "N/A")  # Assuming first provider URL for simplicity

            # Extract skills from description
            skills_found = extract_skills_from_description(description)

            jobs_data.append({
                "title": job_title,
                "company": company,
                "location": location,
                "employment_type": employment_type,
                "description": description[:200],  # Only print a snippet of description
                "url": job_url,
                "skills": skills_found
            })
    else:
        jobs_data.append({"error": "No job listings found or unexpected response format."})

    return jobs_data
