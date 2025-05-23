import pandas as pd
import json
import google.generativeai as genai
from collections import defaultdict
import os
import time
from functools import wraps
import re
from typing import List, Dict, Any, Optional, Tuple

# Configure Gemini API
# It's highly recommended to load your API key from an environment variable for security.
# For example: GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_API_KEY = "GOOGLE_API_KEY"  # change this to your own API key
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash', generation_config={"temperature": 0.0})


def retry_on_error(max_retries=5, delay=1, quota_delay=60):

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if "429" in str(e):
                        print(f"Gemini API quota exceeded, waiting {quota_delay} seconds and retrying...")
                        time.sleep(quota_delay)
                        retries += 1
                        continue
                    if retries == max_retries - 1:
                        print(f"Error after {max_retries} attempts: {e}")
                        return None
                    print(f"Error attempt {retries + 1}: {e}. Retrying in {delay} seconds...")
                    time.sleep(delay)
                    retries += 1
            return None
        return wrapper
    return decorator

def clean_text_response(response_text: str) -> str:
    response_text = re.sub(r'```\s*', '', response_text)
    return response_text.strip()

@retry_on_error(max_retries=3, delay=2)
def merge_field_with_gemini_summary(field_name: str, field_list: List[str]) -> str:
    """
    Uses Gemini to merge a list of text fields (like titles or problems) into a single,
    conise Japanese sentence (max 40 chars) representing the overall theme or issue
    for a given cluster. This is for the *cluster-level summary*.
    """
    # Filter out empty or irrelevant entries (e.g., "related keywords")
    filtered_list = [item for item in field_list if item and not re.search(r'(từ khóa liên quan|関連キーワード|keywords)', item, re.IGNORECASE)]
    if not filtered_list:
        return ""

    prompt = f"""
    Role: You are an expert in analyzing conversational content and need to merge multiple similar text snippets (titles or problems).
    Goal: Create a single, concise, general descriptive sentence (in Japanese) that comprehensively covers all main ideas from the provided list. This sentence should be 40 characters or less. Do NOT phrase it as a question or include question marks. Do NOT include personal information (names, dates, specific amounts, etc.).

    Here is the list of {field_name}s to merge:
    {chr(10).join(filtered_list)}
    """
    try:
        response = model.generate_content(prompt)
        if response.text:
            return response.text.strip()
    except Exception as e:
        print(f"Gemini error merging {field_name} for summary. Retrying:", e)
        time.sleep(2) # Add a small delay before potential retry
    return " ".join(filtered_list) # Fallback: return concatenated strings if Gemini fails

@retry_on_error(max_retries=3, delay=2)
def extract_qa_from_kcs_with_gemini(problem_ja: str, solution_ja: str) -> Optional[str]:
    """
    Uses Gemini to extract the core question from Problem_Ja and a step-by-step
    solution from Solution_Ja that directly and sufficiently answers that core question.
    The solution must ONLY use information from Solution_Ja, without adding or omitting key details.
    """
    if not problem_ja and not solution_ja:
        return None

    # PROMPT: Focus on extracting the CORE question and generating a specific, sufficient solution.
    prompt = f"""
    Role: You are a highly skilled KCS (Knowledge-Centered Service) content analyst specializing in the insurance domain. Your primary goal is to accurately identify the core customer question and extract a precise, step-by-step solution directly from the provided KCS entry.

    Here are examples of how I expect you to process KCS entries:

    Example 1:
    Problem (問題):
    契約者向けのウェブサイトにログインできません。パスワードを何度か入力しましたが、エラーメッセージが表示されます。登録済みのメールアドレスも確認しましたが、問題が解決しません。
    Solution (解決策):
    1. まず、ご使用のブラウザのキャッシュとクッキーをクリアしてください。
    2. 次に、別のブラウザを試すか、シークレットモード（プライベートブラウジング）でログインしてみてください。
    3. それでもログインできない場合は、「パスワードを忘れた方」リンクからパスワードを再設定してください。
    4. これらの手順で解決しない場合は、お客様サポートまでご連絡ください。
    Expected Output:
    question 1: 契約者向けウェブサイトにログインできない場合の対処法は何ですか？
    Solution 1:
        ステップ1: ご使用のブラウザのキャッシュとクッキーをクリアしてください。
        ステップ2: 別のブラウザを試すか、シークレットモードでログインしてみてください。
        ステップ3: それでもログインできない場合は、「パスワードを忘れた方」リンクからパスワードを再設定してください。
        ステップ4: これらの手順で解決しない場合は、お客様サポートまでご連絡ください。

    Example 2:
    Problem (問題):
    自動車保険の契約をオンラインでキャンセルしたいのですが、手続き方法がわかりません。ウェブサイトでキャンセルオプションが見つかりません。
    Solution (解決策):
    1. 当社のウェブサイトにログインし、マイページにアクセスしてください。
    2. 「契約情報」セクションを選択し、「契約変更」をクリックしてください。
    3. 「契約のキャンセル」オプションを選択し、画面の指示に従って手続きを完了してください。
    4. キャンセルには一定の条件と手数料が発生する場合がありますので、ご注意ください。
    Expected Output:
    question 1: 自動車保険の契約をオンラインでキャンセルする方法を教えてください。
    Solution 1:
        ステップ1: 当社のウェブサイトにログインし、マイページにアクセスしてください。
        ステップ2: 「契約情報」セクションを選択し、「契約変更」をクリックしてください。
        ステップ3: 「契約のキャンセル」オプションを選択し、画面の指示に従って手続きを完了してください。
        ステップ4: キャンセルには一定の条件と手数料が発生する場合がありますので、ご注意ください。

    Example 3:
    Problem (問題):
    医療費控除の申請に必要な書類が何か分かりません。どの領収書や明細を提出すればいいですか？
    Solution (解決策):
    1. 医療費控除の対象となるのは、医師の診療費、医薬品費、入院費などです。
    2. 医療費控除に必要な主な書類は、医療費通知、領収書、診断書などです。
    3. 確定申告の際に、これらの書類を税務署に提出する必要があります。
    4. 詳細な情報は国税庁のウェブサイトをご確認ください。
    Expected Output:
    question 1: 医療費控除の申請に必要な書類は何ですか？
    Solution 1:
        ステップ1: 医療費控除の対象となるのは、医師の診療費、医薬品費、入院費などです。
        ステップ2: 医療費控除に必要な主な書類は、医療費通知、領収書、診断書などです。
        ステップ3: 確定申告の際に、これらの書類を税務署に提出する必要があります。
        ステップ4: 詳細な情報は国税庁のウェブサイトをご確認ください。

    Example 4:
    Problem (問題):
    保険料の支払いが遅れてしまいました。どのようにすればよいですか？期限を過ぎてしまったのですが、契約はまだ有効ですか？
    Solution (解決策):
    1. まず、保険契約番号をご確認の上、当社のウェブサイトの支払いページにアクセスしてください。
    2. 延滞料金が発生している場合は、合計金額を支払う必要があります。
    3. 支払い期限を過ぎてからの契約の有効性については、個別の契約条件によって異なりますので、お客様サポートにご連絡いただくか、契約書をご確認ください。
    4. 支払い完了後もご不明な点があれば、当社の担当者にご相談ください。
    Expected Output:
    question 1: 保険料の支払いが遅れた場合の対応策は何ですか？
    Solution 1:
        ステップ1: まず、保険契約番号をご確認の上、当社のウェブサイトの支払いページにアクセスしてください。
        ステップ2: 延滞料金が発生している場合は、合計金額を支払う必要があります。
        ステップ3: 支払い期限を過ぎてからの契約の有効性については、個別の契約条件によって異なりますので、お客様サポートにご連絡いただくか、契約書をご確認ください。
        ステップ4: 支払い完了後もご不明な点があれば、当社の担当者にご相談ください。

    Now, analyze the following KCS entry and generate a similar output:

    Problem (問題):
    {problem_ja}

    Solution (解決策):
    {solution_ja}

    Task:
    1.  **Extract the core, main customer question** from the "Problem" section. This question must be:
        * **Specific, concise, and phrased as a question a customer would actually ask (in Japanese).**
        * It should represent the central inquiry, **not necessarily the entire text** of the "Problem" section.
        * **Prioritize direct questions.** If "Problem" is empty or extremely vague, infer the most likely core question based on what the "Solution" section directly addresses. The inferred question must be what the provided solution *solves*.
        * **Crucially, avoid including any irrelevant details, conversational fillers, or non-question phrasing.**

    2.  **Generate a step-by-step solution** that **directly and exclusively answers the extracted core question.**
        * The solution **MUST ONLY use information available in the "Solution" section provided.** DO NOT add external information or make assumptions.
        * Each step should be **detailed, specific, and actionable**, necessary to solve the core question.
        * **DO NOT include redundant or irrelevant information.** The solution should be concise yet complete for the extracted question.
        * **DO NOT include personal identifiable information (PII)** such as names, specific dates, policy numbers, or exact financial amounts.
        * Begin numbering steps from 'ステップ1:'.

    The output format must be in Japanese as follows:

    question 1: [Extracted Core Customer Question]
    Solution 1:
        ステップ1: [Specific Procedure for the Core Question]
        ステップ2: [Specific Procedure for the Core Question]
        ...
    """
    try:
        response = model.generate_content(prompt)
        if not response.text:
            print("Did not receive response from Gemini for KCS Q&A extraction.")
            return None
        cleaned_response = clean_text_response(response.text)
        return cleaned_response
    except Exception as e:
        print(f"Error processing KCS Q&A with Gemini: {e}")
        raise # Re-raise for retry_on_error

def parse_qa_pairs_from_gemini_response(text: str) -> List[Tuple[str, List[str]]]:
    """
    Parses the Gemini response text into question-answer pairs,
    designed for the specific output format of extract_qa_from_kcs_with_gemini.
    """
    qa_pairs = []
    if not text:
        return qa_pairs

    # Split the entire response by 'question \d+:' to get blocks for each Q&A
    # This regex is robust to variations like "question 1:", "Question 1:", "question1:"
    blocks = re.split(r'question\s*\d+[:：]\s*', text, flags=re.IGNORECASE)

    # The first split part might be empty if the text starts directly with 'question 1:'
    # or it might contain some preamble if Gemini adds it. We'll skip it if empty.
    for i, block in enumerate(blocks):
        if not block.strip():
            continue

        # For each block, the first line after the split is the question, subsequent lines are solution steps
        lines = [line.strip() for line in block.split('\n') if line.strip()]
        if not lines:
            continue

        question = lines[0] # The first line is the extracted core question
        
        steps = []
        solution_section_started = False
        
        # Process the rest of the lines as solution steps
        for line in lines[1:]:
            # Check for "Solution X:" header (case-insensitive, handles both : and ：)
            if re.match(r'Solution\s*\d+[:：]\s*', line, re.IGNORECASE):
                solution_section_started = True
                # If there's content right after "Solution X:", treat it as the first step
                content_after_header = re.sub(r'Solution\s*\d+[:：]\s*', '', line, flags=re.IGNORECASE).strip()
                if content_after_header:
                    steps.append(content_after_header)
                continue
            
            if solution_section_started:
                # If it's a step line (e.g., "ステップ1:", "ステップ2："), extract content
                step_match = re.match(r'ステップ\s*(\d+)[：:]\s*(.*)$', line)
                if step_match:
                    step_content = step_match.group(2).strip()
                    if step_content: # Only add if step content is not empty
                        steps.append(step_content)
                else:
                    # If it's a continuation of the previous step or just general solution text
                    # and not a new step marker, append to the last step if available
                    if steps:
                        steps[-1] += " " + line
                    else: # If no steps yet, but solution section started, treat as the first step
                        steps.append(line)
        
        # Ensure at least one empty step if solution section was indicated but no steps were found
        # This prevents empty solution arrays when Gemini just outputted "Solution 1:" without steps
        if solution_section_started and not steps:
            steps.append("")

        qa_pairs.append((question, steps))
    
    return qa_pairs

def format_kcs_japanese_small_cluster_summary(cluster_title: str, cluster_problem: str, extracted_qa_pairs: List[Tuple[str, List[str]]]) -> str:
    """
    Formats the KCS content for a Small Cluster summary into a structured Japanese KCS string.
    This uses the merged title/problem for the cluster summary and lists the unique extracted Q&A pairs.
    """
    result = []
    result.append(f"１．タイトル：{cluster_title if cluster_title else '（タイトルなし）'}")
    result.append(f"２．課題・質問：{cluster_problem if cluster_problem else '（質問なし）'}")
    result.append("３．解決策：")

    if not extracted_qa_pairs:
        result.append("　（解決策なし）")
        return '\n'.join(result)

    for qidx, (q, steps) in enumerate(extracted_qa_pairs, 1):
        result.append(f"question{qidx}: {q}")
        if not steps:
            result.append(f"　ステップ1：") # Ensure at least a blank step 1 if solution is empty
        else:
            for sidx, step in enumerate(steps, 1):
                result.append(f"　ステップ{sidx}：{step}")
    return '\n'.join(result)

# --- Main Processing Function ---

def process_excel_file(input_path: str, output_path: str):
    """
    Processes an Excel file to:
    1. Group KCS entries by 'Big Cluster' and then by 'Small Cluster'.
    2. For each 'Small Cluster', merge 'Title_Ja' and 'Problem_Ja' using Gemini
        to create a concise cluster-level summary.
    3. For each individual KCS entry within a 'Small Cluster', extract a core question
        from 'Problem_Ja' and a directly answering solution from 'Solution_Ja' using Gemini.
    4. Deduplicate these extracted Q&A pairs based on the core question within each 'Small Cluster'.
    5. Format the combined output into structured KCS documents, with one row per 'Small Cluster'
        in the final Excel file.
    """
    print(f"Đang đọc tệp {input_path}...")
    df = pd.read_excel(input_path)
    df.columns = df.columns.str.strip() # Clean column names
    print(f"Các cột trong tệp: {list(df.columns)}")

    # Check for required columns
    required_cols = ['Title_Ja', 'Problem_Ja', 'Solution_Ja', 'Big Cluster', 'Small Cluster']
    for col in required_cols:
        if col not in df.columns:
            print(f"Thiếu cột bắt buộc: '{col}'. Dừng chương trình.")
            return

    # Filter to only include specified Big Clusters and valid Small Clusters
    allowed_big_clusters =[5,9,12,13,14]
    df_filtered = df[(df['Big Cluster'] != -1) &
                     (df['Small Cluster'] != -1) &
                     (df['Big Cluster'].isin(allowed_big_clusters))].copy() # Use .copy() to avoid SettingWithCopyWarning

    # List to store results for all processed Small Clusters
    all_small_cluster_results = []

    print("\nĐang xử lý các Cụm Nhỏ và trích xuất các cặp Q&A...")

    # Group by both Big Cluster and Small Cluster to process each Small Cluster independently
    # This iterates through unique (Big Cluster, Small Cluster) pairs
    for (big_cluster_id, small_cluster_id), group in df_filtered.groupby(['Big Cluster', 'Small Cluster']):
        print(f"\n--- Đang xử lý Cụm Lớn {big_cluster_id}, Cụm Nhỏ {small_cluster_id} ---")

        # --- Step 1 & 2: Merge Title_Ja and Problem_Ja for the CURRENT Small Cluster ---
        # Collect all titles and problems from KCS entries within this specific Small Cluster
        all_titles_in_small_cluster = group['Title_Ja'].dropna().astype(str).tolist()
        all_problems_in_small_cluster = group['Problem_Ja'].dropna().astype(str).tolist()

        # Use Gemini to merge these for the Small Cluster's summary
        merged_small_cluster_title = merge_field_with_gemini_summary("title", all_titles_in_small_cluster)
        time.sleep(1) # Add a small delay between API calls
        merged_small_cluster_problem = merge_field_with_gemini_summary("problem", all_problems_in_small_cluster)
        time.sleep(1) # Add a small delay between API calls

        # --- Step 3 & 4: Extract Core Q&A and Deduplicate within the CURRENT Small Cluster ---
        unique_extracted_qa_pairs_for_small_cluster = []
        unique_extracted_questions_set_for_small_cluster = set() #

        # Iterate through each individual KCS entry within the current Small Cluster
        for index, row in group.iterrows():
            original_title = str(row['Title_Ja']).strip() if pd.notna(row['Title_Ja']) else ""
            original_problem = str(row['Problem_Ja']).strip() if pd.notna(row['Problem_Ja']) else ""
            original_solution = str(row['Solution_Ja']).strip() if pd.notna(row['Solution_Ja']) else ""

            if not original_problem and not original_solution:
                continue 
            gemini_qa_response = extract_qa_from_kcs_with_gemini(original_problem, original_solution)

            if gemini_qa_response:
                parsed_qa = parse_qa_pairs_from_gemini_response(gemini_qa_response)
                
                if parsed_qa:
                    extracted_core_question = parsed_qa[0][0].strip() # Get the first (and usually only) extracted core question
                    extracted_solution_steps = parsed_qa[0][1] # Get its corresponding steps

                    if extracted_core_question and extracted_core_question not in unique_extracted_questions_set_for_small_cluster:
                        unique_extracted_questions_set_for_small_cluster.add(extracted_core_question)
                        unique_extracted_qa_pairs_for_small_cluster.append((extracted_core_question, extracted_solution_steps))

        # After processing all KCS entries in this Small Cluster, format its summary
        formatted_kcs_jp_small_cluster_summary = format_kcs_japanese_small_cluster_summary(
            merged_small_cluster_title,
            merged_small_cluster_problem,
            unique_extracted_qa_pairs_for_small_cluster
        )

        # Create a JSON representation for the Small Cluster's summary
        # CONVERT int64 TO int TO AVOID JSON SERIALIZATION ERROR
        merged_small_cluster_json = json.dumps({
            "big_cluster_id": int(big_cluster_id),     # FIX: Convert numpy.int64 to standard int
            "small_cluster_id": int(small_cluster_id), # FIX: Convert numpy.int64 to standard int
            "merged_small_cluster_title": merged_small_cluster_title,
            "merged_small_cluster_problem": merged_small_cluster_problem,
            "extracted_qa_pairs": [{"question": q, "solution_steps": s} for q, s in unique_extracted_qa_pairs_for_small_cluster]
        }, ensure_ascii=False, indent=2)

        # Add the result of this Small Cluster to the overall list
        all_small_cluster_results.append({
            "Big_Cluster_ID": big_cluster_id, # Keep original type here for Excel column if needed, or convert with int()
            "Small_Cluster_ID": small_cluster_id, # Keep original type here for Excel column if needed, or convert with int()
            "Merged_Small_Cluster_Title_Ja": merged_small_cluster_title,
            "Merged_Small_Cluster_Problem_Ja": merged_small_cluster_problem,
            "Extracted_Q_A_Pairs_Count": len(unique_extracted_qa_pairs_for_small_cluster),
            "Formatted_KCS_JP_Small_Cluster_Summary": formatted_kcs_jp_small_cluster_summary,
            "Merged_Small_Cluster_JSON": merged_small_cluster_json
        })
        print(f"--- Đã hoàn thành Cụm Lớn {big_cluster_id}, Cụm Nhỏ {small_cluster_id} ---")

    # Save all collected Small Cluster summaries to a single Excel sheet
    print(f"\nĐang lưu kết quả vào tệp {output_path}...")
    df_output = pd.DataFrame(all_small_cluster_results)
    df_output.to_excel(output_path, index=False, sheet_name='Small_Cluster_Summaries')
    print("Xử lý KCS hoàn tất! Mỗi Cụm Nhỏ được tóm tắt trong một hàng.")


if __name__ == "__main__":
    # --- Main Execution Block ---

    # Define input and output file paths
    # IMPORTANT: Adjust this path to your actual Excel file location
    input_file = 'D:/KCS_Clustering/23_05/KCS_Clustering_23052025_updated.xlsx' # input file HDBSCAN result
    output_file = 'kcs_small_cluster_23_05_2025_final_cluster_summary_cluster_5_9_12_13_14.xlsx' # Output file 

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Run the processing
    process_excel_file(input_file, output_file)