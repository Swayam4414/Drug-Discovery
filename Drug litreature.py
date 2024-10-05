import os
import sys
import time
import re
import requests
import nltk
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from bs4 import BeautifulSoup
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai  # Add this to interact with the ChatGPT API

# Initialize NLTK
nltk.download('punkt')

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("chatbot.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger()

# Constants and Configuration
TOKEN_LIMIT = 2000  # Token limit per summary (adjust as needed)
NCBI_API_KEY = os.getenv('NCBI_API_KEY')

if not NCBI_API_KEY:
    logger.error("Error: NCBI_API_KEY environment variable not set.")
    sys.exit(1)

# Initialize a persistent session
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/115.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/"
})

# Set OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Make sure this is set in your environment

if not OPENAI_API_KEY:
    logger.error("Error: OPENAI_API_KEY environment variable not set.")
    sys.exit(1)

openai.api_key = OPENAI_API_KEY

# Alternative titles for different sections
alternative_titles = {
    'Abstract': ['abstract'],
    'Introduction': ['introduction', 'introductory remarks'],
    'Background': ['background', 'related work', 'literature review', 'previous studies'],
    'Literature Review': ['literature review', 'related work'],
    'Methodology': ['methodology', 'materials and methods', 'methods', 'experimental procedures'],
    'Findings': ['findings', 'results', 'outcomes', 'data analysis'],
    'Discussion': ['discussion', 'interpretation of results'],
    'Limitations': ['limitations', 'constraints', 'limitations and delimitations'],
    'Future Work': ['future work', 'recommendations', 'directions for future research'],
    'Ethical Considerations': ['ethical considerations', 'ethics statement'],
    'Conclusion': ['conclusion', 'final thoughts', 'concluding remarks'],
    'Acknowledgments': ['acknowledgments', 'acknowledgements']
}


def search_pmc(query, max_results=10):
    """
    Search PubMed Central (PMC) for articles related to the query.
    """
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        'db': 'pmc',
        'term': query,
        'retmax': max_results,
        'api_key': NCBI_API_KEY,
        'retmode': 'json'
    }
    try:
        logger.info(f"Searching PMC for query: '{query}' with max_results={max_results}")
        response = session.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        pmc_ids = data.get('esearchresult', {}).get('idlist', [])
        logger.info(f"Found {len(pmc_ids)} articles.")
        return pmc_ids
    except Exception as e:
        logger.error(f"Error searching PMC: {e}")
        return []


def fetch_article_details(pmc_ids):
    """
    Fetch article details (title and DOI) from PMC IDs.
    """
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    params = {
        'db': 'pmc',
        'id': ",".join(pmc_ids),
        'api_key': NCBI_API_KEY,
        'retmode': 'json'
    }
    try:
        logger.info("Fetching article details from PMC IDs...")
        response = session.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        articles = []
        for pmc_id in pmc_ids:
            doc = data.get('result', {}).get(pmc_id, {})
            title = doc.get('title', 'No Title')
            doi_list = doc.get('articleids', [])
            doi_value = next((item['value'] for item in doi_list if item['idtype'] == 'doi'), 'No DOI')
            articles.append({'pmc_id': pmc_id, 'title': title, 'doi': doi_value})
        logger.info(f"Retrieved details for {len(articles)} articles.")
        return articles
    except Exception as e:
        logger.error(f"Error fetching article details: {e}")
        return []


def display_articles(articles):
    """
    Display a list of articles with titles and DOIs.
    """
    logger.info("\n=== Search Results ===\n")
    for idx, article in enumerate(articles, 1):
        print(f"{idx}. Title: {article['title']}")
        print(f"   DOI: {article['doi']}\n")


def get_paper_url_from_doi(doi):
    """
    Use the DOI to get the final research paper URL by accessing the DOI redirect service.
    """
    doi_url = f"https://doi.org/{doi}"
    try:
        logger.info(f"Accessing DOI URL: {doi_url}")
        response = session.get(doi_url, allow_redirects=True, timeout=10)
        if response.status_code == 200:
            logger.info(f"Retrieved paper URL: {response.url}")
            return response.url  # Final redirected URL
        elif response.status_code == 403:
            logger.warning(f"Access to the paper is forbidden (403). It might be behind a paywall.")
            return None
        else:
            logger.error(f"Error accessing DOI: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Error fetching paper from DOI: {e}")
        return None


def fetch_paper_webpage(paper_url):
    """
    Fetch the webpage of the research paper using its final URL.
    """
    try:
        logger.info(f"Fetching the research paper webpage: {paper_url}")
        response = session.get(paper_url, timeout=15)
        if response.status_code == 200:
            logger.info("Successfully fetched the paper webpage.")
            return response.text  # Return the HTML content of the webpage
        elif response.status_code == 403:
            logger.warning(f"Access to the paper webpage is forbidden (403). It might be behind a paywall.")
            return None
        else:
            logger.error(f"Error fetching paper webpage: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Error fetching paper webpage: {e}")
        return None


def extract_sections_from_webpage(html_content):
    """
    Extract sections like Abstract, Introduction, etc., from the webpage.
    """
    soup = BeautifulSoup(html_content, 'html.parser')

    # Initialize sections dictionary
    sections = {
        'Abstract': '',
        'Introduction': '',
        'Background': '',
        'Literature Review': '',
        'Methodology': '',
        'Findings': '',
        'Discussion': '',
        'Conclusion': ''
    }

    # Combine all possible section headers
    all_headers = []
    for section, keywords in alternative_titles.items():
        for keyword in keywords:
            all_headers.append(keyword.lower())

    # Find all headers in the HTML
    headers = soup.find_all(re.compile('^h[1-6]$'))
    for header in headers:
        header_text = header.get_text().strip().lower()
        for section, keywords in alternative_titles.items():
            if any(keyword in header_text for keyword in keywords):
                # Extract content until the next header
                content = []
                for sibling in header.next_siblings:
                    # Stop when reaching the next header or a non-text element
                    if sibling.name and re.match('^h[1-6]$', sibling.name):
                        break
                    # Skip images and figure elements
                    if sibling.name in ['img', 'figure', 'figcaption']:
                        logger.info(f"Ignoring section '{section}' as it contains images.")
                        content = []  # Clear content to ignore the whole section
                        break
                    # Only include paragraph text content
                    if sibling.name == 'p':
                        content.append(sibling.get_text(separator=" ", strip=True))
                # If no images were found, save the content
                if content:
                    sections[section] = " ".join(content)
                break  # Move to the next header after finding a match

    # Remove empty sections
    sections = {k: v for k, v in sections.items() if v}

    if not sections:
        logger.warning("No recognizable sections found.")
    else:
        logger.info(f"Identified sections: {', '.join(sections.keys())}")

    return sections


def simplify_text(text, section_name):
    """
    Simplify text using the ChatGPT model from OpenAI's API.
    """
    prompt = (
        f"Summarize the following text from the {section_name} section of a research paper. "
        f"Provide a detailed, data-driven summary focusing on specific content, methodologies, findings, and conclusions. "
        f"Include references to the research data and avoid generic statements.\n\nText:\n{text}\n\nSummary:"
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # You can use "gpt-4" or other models if available
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes academic research."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=TOKEN_LIMIT,
            n=1,
            stop=None,
            temperature=0.7,
        )

        summary = response['choices'][0]['message']['content']
        return summary.strip()
    except Exception as e:
        logger.error(f"Error generating summary with ChatGPT: {e}")
        return ""


def split_into_chunks(text, token_limit=512, special_token_buffer=2):
    """
    Split text into chunks without splitting sentences, ensuring that no chunk exceeds the token limit.
    """
    adjusted_token_limit = token_limit - special_token_buffer
    sentences = sent_tokenize(text, language='english')  # Tokenize text into sentences
    chunks = []
    current_chunk = ""
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = len(sentence.split())  # Count tokens as words
        if current_tokens + sentence_tokens <= adjusted_token_limit:
            current_chunk += " " + sentence
            current_tokens += sentence_tokens
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence  # Start a new chunk with the current sentence
            current_tokens = sentence_tokens

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def combine_summaries(summaries):
    """
    Combine simplified chunks into a coherent summary.
    """
    return "\n\n".join(summaries)


def summarize_sections(sections):
    """
    Summarize each extracted section using ChatGPT API.
    """
    detailed_summary = ""

    section_order = [
        'Abstract',
        'Introduction',
        'Background',
        'Literature Review',
        'Methodology',
        'Findings',
        'Discussion',
        'Conclusion'
    ]

    for sec in section_order:
        section_text = sections.get(sec, "")
        if section_text:
            try:
                logger.info(f"Simplifying section: {sec}")
                chunks = split_into_chunks(section_text)
                simplified_sections = []
                for chunk in tqdm(chunks, desc=f"Simplifying {sec}"):
                    simplified = simplify_text(chunk, sec)
                    if simplified:
                        simplified_sections.append(simplified)
                    else:
                        simplified_sections.append(chunk)  # Fallback to original if simplification fails
                    time.sleep(0.5)  # Adjust based on model inference speed
                section_summary = combine_summaries(simplified_sections)
                detailed_summary += f"### {sec}\n\n{section_summary}\n\n"
            except Exception as e:
                logger.error(f"Error summarizing section '{sec}': {e}")
                detailed_summary += f"### {sec}\n\nError summarizing this section.\n\n"
        else:
            logger.info(f"Section '{sec}' not found or empty.")

    return f"**Detailed Summary of Research Paper:**\n\n{detailed_summary}"


def main():
    print("=== AI in Healthcare Research Paper Chatbot ===\n")
    field = input("Enter the name of the field (e.g., 'AI in healthcare'): ").strip()
    if not field:
        logger.error("No field entered. Exiting.")
        print("No field entered. Exiting.")
        return

    pmc_ids = search_pmc(field, max_results=10)
    if not pmc_ids:
        logger.error("No articles found. Exiting.")
        print("No articles found. Exiting.")
        return

    articles = fetch_article_details(pmc_ids)
    if not articles:
        logger.error("Failed to retrieve article details. Exiting.")
        print("Failed to retrieve article details. Exiting.")
        return

    display_articles(articles)

    while True:
        try:
            selection = int(input(f"\nSelect a paper to summarize (1-{len(articles)}): "))
            if 1 <= selection <= len(articles):
                selected_article = articles[selection - 1]
                break
            else:
                print(f"Please enter a number between 1 and {len(articles)}.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")

    doi = selected_article['doi']
    if doi == 'No DOI':
        logger.error("Selected article does not have a DOI. Exiting.")
        print("Selected article does not have a DOI. Exiting.")
        return

    paper_url = get_paper_url_from_doi(doi)
    if not paper_url:
        logger.error("Failed to retrieve the paper URL. Exiting.")
        print("Failed to retrieve the paper URL. Exiting.")
        return

    print(f"\nResearch Paper URL: {paper_url}")

    html_content = fetch_paper_webpage(paper_url)
    if not html_content:
        logger.error("Failed to fetch the paper webpage. Exiting.")
        print("Failed to fetch the paper webpage. Exiting.")
        return

    sections = extract_sections_from_webpage(html_content)
    if not sections:
        logger.error("Failed to extract any sections from the webpage. Exiting.")
        print("Failed to extract any sections from the webpage. Exiting.")
        return

    formatted_summary = summarize_sections(sections)

    print("\n=== Detailed Summary ===\n")
    print(formatted_summary)

    save_option = input("\nDo you want to save the summary to a text file? (yes/no): ").strip().lower()
    if save_option in ['yes', 'y']:
        sanitized_doi = re.sub(r'[^\w\-_. ]', '_', doi)
        filename = f"summary_{sanitized_doi}.txt"
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(formatted_summary)
            logger.info(f"Summary saved to {filename}")
            print(f"Summary saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving summary to file: {e}")
            print(f"Error saving summary to file: {e}")


if __name__ == "__main__":
    main()
