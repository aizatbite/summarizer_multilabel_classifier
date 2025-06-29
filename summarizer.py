#24 April 2025 | Summarize articles from an Excel file using Gemini API

import pandas as pd
import google.generativeai as genai
import time
import os
import re
from tqdm import tqdm


# Configure the Gemini API
def setup_gemini(api_key):
    genai.configure(api_key=api_key)
    # Initialize the model - choose model
    model = genai.GenerativeModel('gemini-2.0-flash')
    return model


def summarize_articles(input_file, output_file, api_key, evaluation_prompt, summary_prompt):
    """
    Summarize articles from an Excel file using Gemini API

    Parameters:
    - input_file: path to input Excel file with articles
    - output_file: path to save the output Excel file with summaries
    - api_key: Gemini API key
    - evaluation_prompt: The prompt template to evaluate article relevance
    - summary_prompt: The prompt template to use for summarization
    """
    # Setup Gemini
    model = setup_gemini(api_key)

    # Read the Excel file
    df = pd.read_excel(input_file)

    # Print column names to debug
    print("Excel file columns:", df.columns.tolist())

    # Add new columns for results
    df['relevant'] = ''
    df['topic_category'] = ''  # For "LTE/5G private networks", "fixed wireless access", or "network slicing"
    df['news_category'] = ''  # For the type of news (investments, rollouts, etc.)
    df['reason'] = ''  # column for the reason
    df['summary'] = ''
    df['tech_mentions'] = ''  # column for technology mentions

    # Define expanded technology keywords with categories
    tech_mapping = {
        'IoT': ['IoT', 'Internet of Things', 'sensor', 'sensors', 'smart device', 'connected device'],
        'edge computing': ['edge computing', 'edge AI', 'edge analytics'],
        'network slicing': ['network slicing', 'network slice'],
        'AR': ['AR', 'augmented reality'],
        'VR': ['VR', 'virtual reality'],
        'robots/drones': ['robot', 'robots', 'robotics', 'drone', 'drones', 'autonomous vehicle', 'autonomous machines'],
        'AI/ML': ['AI', 'artificial intelligence', 'machine learning', 'ML', 'deep learning', 'neural network']
    }

    # Process each article
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing articles"):
        # Get content from column B directly
        if len(df.columns) > 1:
            content = str(df.iloc[i, 1])
            print(f"Article {i + 1} content length: {len(content)}")
        else:
            print(f"Warning: Excel file has fewer than 2 columns at row {i + 1}")
            continue

        # Skip if content is empty or too short
        if len(content) < 10:
            print(f"Warning: Content for article {i + 1} is too short or empty. Skipping.")
            df.at[i, 'relevant'] = "N"
            df.at[i, 'topic_category'] = "N/A"
            df.at[i, 'news_category'] = "N/A"
            df.at[i, 'reason'] = "Content too short or empty"
            df.at[i, 'summary'] = "Error: Content too short or empty"
            df.at[i, 'tech_mentions'] = "N/A"
            continue

        try:
            # STEP 1: Evaluate relevance
            eval_prompt = evaluation_prompt.format(content=content)
            evaluation_response = model.generate_content(eval_prompt)
            evaluation_result = evaluation_response.text.strip()

            print(f"Evaluation result for article {i + 1}:\n{evaluation_result}")

            # Extract relevance (Y/N)
            relevance_match = re.search(r'Relevance:\s*([YN])', evaluation_result)
            relevant = relevance_match.group(1) if relevance_match else 'N'

            # Default values
            topic_category = "N/A"
            news_category = "N/A"
            tech_mentions = "None"  # Default for tech mentions
            reason = "No reason provided"  # Default for reason

            # Extract reason using regex
            reason_match = re.search(r'Reason:\s*(.*?)(?:\n|$)', evaluation_result)
            if reason_match:
                reason = reason_match.group(1).strip()

            # If relevant, extract categories
            if relevant == 'Y':
                # Extract topic category
                topic_match = re.search(r'Topic Category:\s*(.*?)(?:\n|$)', evaluation_result)
                if topic_match:
                    topic = topic_match.group(1).strip()
                    if "private network" in topic.lower():
                        topic_category = "LTE/5G private networks"
                    elif "fixed wireless" in topic.lower():
                        topic_category = "fixed wireless access"
                    elif "network slicing" in topic.lower():
                        topic_category = "network slicing"

                # Extract news category
                news_match = re.search(r'News Category:\s*(.*?)(?:\n|$)', evaluation_result)
                if news_match:
                    news_category = news_match.group(1).strip()

            # Store the evaluation results
            df.at[i, 'relevant'] = relevant
            df.at[i, 'topic_category'] = topic_category
            df.at[i, 'news_category'] = news_category
            df.at[i, 'reason'] = reason  # Store the reason

            # STEP 2: Summarize only if relevant
            if relevant == 'Y':
                summary_full_prompt = summary_prompt.format(content=content)
                summary_response = model.generate_content(summary_full_prompt)
                summary_text = summary_response.text

                # Scan the original content directly for technology mentions
                found_techs = set()

                # Check for technologies in the original content
                for category, keywords in tech_mapping.items():
                    for keyword in keywords:
                        # Look in original content
                        if re.search(r'\b' + re.escape(keyword) + r'\b', content, re.IGNORECASE):
                            found_techs.add(category)
                            break  # Once we find one keyword in a category, we can move to the next category

                if found_techs:
                    tech_mentions = ", ".join(sorted(found_techs))

                df.at[i, 'summary'] = summary_text
                df.at[i, 'tech_mentions'] = tech_mentions
            else:
                df.at[i, 'summary'] = "Not relevant for summarization"
                df.at[i, 'tech_mentions'] = "N/A"

            # Add a small delay to avoid hitting rate limits
            time.sleep(1)

        except Exception as e:
            print(f"Error processing article {i + 1}: {str(e)}")
            df.at[i, 'relevant'] = "Error"
            df.at[i, 'topic_category'] = "Error"
            df.at[i, 'news_category'] = "Error"
            df.at[i, 'reason'] = f"Error: {str(e)}"
            df.at[i, 'summary'] = f"Error: {str(e)}"
            df.at[i, 'tech_mentions'] = "Error"

    # Make sure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save the results
    df.to_excel(output_file, index=False)
    print(f"Results saved to {output_file}")

    # Print a summary of the results
    relevance_counts = df['relevant'].value_counts()
    print("\nRelevance counts:")
    print(relevance_counts)

    if 'Y' in relevance_counts:
        print("\nTopic categories for relevant articles:")
        print(df[df['relevant'] == 'Y']['topic_category'].value_counts())

        print("\nNews categories for relevant articles:")
        print(df[df['relevant'] == 'Y']['news_category'].value_counts())

        print("\nTech mentions for relevant articles:")
        print(df[df['relevant'] == 'Y']['tech_mentions'].value_counts())


if __name__ == "__main__":
    # Configuration
    API_KEY = "XXX"  # Insert API key
    INPUT_FILE = r"XXX"  # Excel file with article text in Column B
    OUTPUT_FILE = r"XXX"

    # Evaluation prompt
    EVALUATION_PROMPT = """
    As an Enterprise 5G and LTE networks market analyst, evaluate whether the given article is relevant using a three-step process:

    STEP 1: TOPIC RELEVANCE
    First, determine if the article's primary focus is on one of these core topics:
    - 5G/LTE private networks for businesses and enterprises only
    - Fixed wireless access (FWA) for businesses and enterprises only
    - Network slicing for businesses and enterprises only

    If the article is NOT primarily about one of these three core topics, immediately classify it as "N" and stop analysis.

    STEP 2: CONTENT TYPE ASSESSMENT
    If the article passes Step 1, evaluate if it specifically covers any of these content types related to the core topics:
    - Investment or merger/acquisition
    - Network rollout
    - Alliance/Sales partnership
    - Product/service launch
    - Spectrum access
    - Test/Trials
    - R&D
    - Contract win
    - Certification or technology integration

    STEP 3: EXCLUSION CRITERIA
    Even if the article meets the criteria in Steps 1 and 2, classify it as "N" if it falls into any of these categories:
    - Articles containing analyst opinions or from analyst firms (e.g., Gartner, IDC)
    - Promotional or marketing content
    - Research-type or analysis articles that primarily discuss market trends, potential use cases, or industry insights without announcing specific news events
    - Opinion pieces, thought leadership articles, or "state of the industry" content
    - Articles that simply discuss potential opportunities or benefits without announcing specific deployments, partnerships, or product launches
    - Event announcements, webinar promotions, or virtual conference invitations
    - Workshop, seminar, or industry event promotions

    MANDATORY EXCLUSIONS:
    - Articles that are primarily research, analysis, or opinion-based → "N"
    - Articles discussing general market trends without specific news events → "N"
    - Virtual event announcements, conference promotions, or webinar invitations → "N"

    OUTPUT FORMAT:
    1. Relevance: "Y" or "N"
    2. If "Y", specify which core topic: "LTE/5G private networks" or "Fixed wireless access" or "Network slicing"
    3. If "Y", specify which content type: [one of the nine categories listed above]

    IMPORTANT: The article must first be primarily about the core topics and must be announcing a specific news event. Articles that primarily discuss trends, potential use cases, "why X matters", or promote industry events are NOT relevant.

    Please format your response like this:
    Relevance: [Y/N]
    Topic Category: [only if Y: "LTE/5G private networks" or "fixed wireless access" or "network slicing"]
    News Category: [only if Y: one of the event categories]
    Reason: [brief explanation of your decision]

    Here is the text to evaluate:
    {content}
    """


    # summarization prompt template
    SUMMARY_PROMPT = """
    As a Enterprise 5G and LTE networks market analyst, summarize the provided text while following these guidelines:
    1. Create a detailed yet concise summary (max 100 words) in a single paragraph.
    2. Focus on information related to LTE/5G private networks, fixed wireless access, or network slicing, and the organizations involved.
    3. Rely exclusively on the provided text without adding external information.
    4. Remove unnecessary adjectives, promotional terms, and extraneous language.
    5. The summary should be self-contained without any separate sections or labels.
    6. The summary should exclude all sentences or statements that promote the companies like its customers, achievements. 
    
    Context: As a market analyst, you own a market tracker to monitor industry developments in LTE and 5G private networks, fixed wireless access, and network slicing. The types of news you will be interested in are related to investments or mergers/acquisitions, network rollouts, partnership, product/service launches, spectrum access, trials, R&D, contract wins, and certification or technology integrations.
    
    Example format:
    [Main summary paragraph]

    Text to summarize:
    {content}
    """
    # Run the summarization
    summarize_articles(INPUT_FILE, OUTPUT_FILE, API_KEY, EVALUATION_PROMPT, SUMMARY_PROMPT)