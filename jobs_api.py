import http.client
import json

# Function to fetch jobs from the API
def fetch_jobs(query, location="United States", employment_types="fulltime;parttime;intern;contractor", remote_only=False):
    """
    Fetch job listings from the Jobs API based on the provided parameters.

    Args:
        query (str): Job title or category to search for (e.g., 'Web Developer').
        location (str): Location to search for jobs (default: 'United States').
        employment_types (str): Employment types (e.g., 'fulltime;parttime;intern;contractor').
        remote_only (bool): Whether to include only remote jobs (default: False).

    Returns:
        dict: Parsed JSON response containing job listings.
    """
    # Establish API connection
    conn = http.client.HTTPSConnection("jobs-api14.p.rapidapi.com")

    # API headers
    headers = {
        'x-rapidapi-key': "63e17585b1mshf1efd542d30ab4dp14bb65jsn742d404f66d3",
        'x-rapidapi-host': "jobs-api14.p.rapidapi.com"
    }

    # Build API endpoint with query parameters
    endpoint = f"/v2/list?query={query}&location={location}&autoTranslateLocation=false&remoteOnly={str(remote_only).lower()}&employmentTypes={employment_types}"

    try:
        # Make API request
        conn.request("GET", endpoint, headers=headers)
        res = conn.getresponse()
        data = res.read()

        # Parse and return JSON response
        return json.loads(data.decode("utf-8"))

    except Exception as e:
        print(f"Error fetching jobs: {e}")
        return {"error": str(e)}

    finally:
        conn.close()


# Example usage
if __name__ == "__main__":
    # Example: Fetch jobs for "Data Scientist"
    query = "Data Scientist"
    job_results = fetch_jobs(query=query, location="United States", employment_types="fulltime", remote_only=True)

    # Print fetched job results
    if "jobs" in job_results:
        for job in job_results["jobs"]:
            print(f"Job Title: {job.get('title', 'N/A')}")
            print(f"Company: {job.get('companyName', 'N/A')}")
            print(f"Location: {job.get('location', 'N/A')}")
            print(f"Job Link: {job.get('jobUrl', '#')}")
            print("---")
    else:
        print("No jobs found or an error occurred.")
