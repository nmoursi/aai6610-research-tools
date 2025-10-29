import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from openai import OpenAI
import re
from pathlib import Path

class PaperClusterer:
    def __init__(self, api_key):
        """Initialize with OpenAI API key"""
        self.client = OpenAI(api_key=api_key)

    def parse_papers(self, file_path):
        """Parse Arxiv-like paper metadata blocks from text file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()

    # Split each paper block by separator line
        blocks = re.split(r'-{5,}', text)
        papers = []

        for block in blocks:
            lines = [line.strip() for line in block.splitlines() if line.strip()]
            if not lines:
                continue

        # Extract TITLE and ABSTRACT
            title_match = re.search(r'^TITLE:\s*(.*)', block, re.MULTILINE)
            abstract_match = re.search(r'^ABSTRACT:\s*(.*)', block, re.MULTILINE | re.DOTALL)

            title = title_match.group(1).strip() if title_match else None
            abstract = abstract_match.group(1).strip() if abstract_match else None

            if title and abstract and len(abstract.split()) > 10:
                papers.append({
                    'title': title,
                    'content': abstract
                })

        print(f"✓ Found {len(papers)} papers")
        return papers
    
    def generate_embeddings(self, texts, batch_size=100):
        """Generate embeddings using OpenAI API with batching"""
        print(f"\nGenerating embeddings for {len(texts)} papers...")
        all_embeddings = []
        
        # Process in batches to avoid rate limits
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            print(f"  Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}...")
            
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=batch
            )
            
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        
        embeddings = np.array(all_embeddings)
        print(f"✓ Embeddings generated: {embeddings.shape}")
        return embeddings
    
    def cluster_papers(self, embeddings, k_values=None):
        """Perform k-means clustering for different k values"""
        if k_values is None:
            # For 40 papers, test more cluster options
            k_values = [3, 4, 5, 6, 7, 8]
        
        results = []
        print(f"\n{'='*60}")
        print("CLUSTERING ANALYSIS")
        print('='*60)
        
        for k in k_values:
            print(f"\nClustering with k={k}...")
            
            # Perform k-means clustering
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)
            
            # Calculate silhouette score
            silhouette = silhouette_score(embeddings, labels)
            
            # Count papers per cluster
            unique, counts = np.unique(labels, return_counts=True)
            cluster_sizes = dict(zip(unique, counts))
            
            results.append({
                'k': k,
                'labels': labels,
                'silhouette': silhouette,
                'centroids': kmeans.cluster_centers_,
                'cluster_sizes': cluster_sizes
            })
            
            print(f"  Silhouette score: {silhouette:.4f}")
            print(f"  Cluster sizes: {cluster_sizes}")
        
        return results
    
    def analyze_clusters(self, papers, clustering_results):
        """Analyze and display clustering results"""
        print("\n" + "="*80)
        print("DETAILED CLUSTERING RESULTS")
        print("="*80)
        
        # Find best k based on silhouette score
        best_result = max(clustering_results, key=lambda x: x['silhouette'])
        print(f"\n*** BEST CONFIGURATION: k={best_result['k']} clusters ***")
        print(f"*** Silhouette Score: {best_result['silhouette']:.4f} ***\n")
        
        # Display results for each k
        for result in clustering_results:
            k = result['k']
            labels = result['labels']
            silhouette = result['silhouette']
            cluster_sizes = result['cluster_sizes']
            
            print(f"\n{'='*80}")
            print(f"k = {k} clusters | Silhouette = {silhouette:.4f}")
            print('='*80)
            
            # Group papers by cluster
            clusters = {}
            for idx, label in enumerate(labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(papers[idx])
            
            # Display each cluster
            for cluster_id in sorted(clusters.keys()):
                papers_in_cluster = clusters[cluster_id]
                print(f"\n--- Cluster {cluster_id + 1} ({len(papers_in_cluster)} papers) ---")
                for paper in papers_in_cluster:
                    # Truncate long titles
                    title = paper['title']
                    if len(title) > 100:
                        title = title[:97] + "..."
                    print(f"  • {title}")
        
        return best_result
    
    def visualize_clusters(self, embeddings, labels, papers, output_file='clusters_visualization.png'):
        """Create t-SNE visualization of clusters"""
        print(f"\nCreating t-SNE visualization...")
        
        # Reduce to 2D using t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # Create plot
        plt.figure(figsize=(14, 10))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                            c=labels, cmap='tab10', s=100, alpha=0.6, edgecolors='black')
        
        # Add labels for each point
        for i, paper in enumerate(papers):
            # Use first 30 chars of title
            short_title = paper['title'][:30] + "..." if len(paper['title']) > 30 else paper['title']
            plt.annotate(f"{i}", (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                        fontsize=7, alpha=0.7)
        
        plt.colorbar(scatter, label='Cluster')
        plt.title(f'Paper Clustering Visualization (k={len(np.unique(labels))} clusters)', fontsize=16)
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Visualization saved to: {output_file}")
        plt.close()
    
    def save_results(self, papers, clustering_results, output_file='clustering_results.txt'):
        """Save clustering results to file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("PAPER CLUSTERING ANALYSIS RESULTS\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Total papers analyzed: {len(papers)}\n\n")
            
            # Find best configuration
            best_result = max(clustering_results, key=lambda x: x['silhouette'])
            f.write(f"BEST CONFIGURATION: k={best_result['k']} clusters\n")
            f.write(f"Silhouette Score: {best_result['silhouette']:.4f}\n\n")
            
            # Write summary of all k values
            f.write("Summary of all clustering attempts:\n")
            f.write("-" * 80 + "\n")
            for result in clustering_results:
                f.write(f"k={result['k']}: Silhouette={result['silhouette']:.4f}, ")
                f.write(f"Sizes={result['cluster_sizes']}\n")
            
            # Write detailed results
            for result in clustering_results:
                k = result['k']
                labels = result['labels']
                silhouette = result['silhouette']
                
                f.write(f"\n\n{'='*80}\n")
                f.write(f"k = {k} clusters | Silhouette Score = {silhouette:.4f}\n")
                f.write('='*80 + "\n")
                
                # Group papers by cluster
                clusters = {}
                for idx, label in enumerate(labels):
                    if label not in clusters:
                        clusters[label] = []
                    clusters[label].append(papers[idx])
                
                # Write each cluster
                for cluster_id in sorted(clusters.keys()):
                    papers_in_cluster = clusters[cluster_id]
                    f.write(f"\n--- Cluster {cluster_id + 1} ({len(papers_in_cluster)} papers) ---\n")
                    for paper in papers_in_cluster:
                        f.write(f"  • {paper['title']}\n")
                    f.write("\n")
        
        print(f"\n✓ Results saved to: {output_file}")
    
    def run_full_analysis(self, input_file, output_dir="arxiv_clustering_result", 
                     output_file="clustering_results.txt", k_values=None, visualize=True):
        """Run complete clustering analysis pipeline and save outputs in a folder"""
        print("="*80)
        print("PAPER CLUSTERING ANALYSIS")
        print("="*80)
        print(f"\nInput file: {input_file}")

    # Step 0: Create output directory
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 All outputs will be saved in: {output_dir}")

    # Step 1: Parse papers
        papers = self.parse_papers(input_file)

    # Step 2: Generate embeddings
        texts = [paper['content'] for paper in papers]
        embeddings = self.generate_embeddings(texts)

    # Step 3: Perform clustering
        clustering_results = self.cluster_papers(embeddings, k_values)

    # Step 4: Analyze and display results
        best_result = self.analyze_clusters(papers, clustering_results)

    # Step 5: Visualize (optional)
        if visualize:
            viz_path = os.path.join(output_dir, "clusters_visualization.png")
            self.visualize_clusters(embeddings, best_result['labels'], papers, output_file=viz_path)

    # Step 6: Save results text file
        results_path = os.path.join(output_dir, output_file)
        self.save_results(papers, clustering_results, output_file=results_path)

    # Step 7: Save embeddings
        emb_path = os.path.join(output_dir, "paper_embeddings.npy")
        np.save(emb_path, embeddings)

        print(f"✓ Results saved to: {results_path}")
        print(f"✓ Visualization saved to: {viz_path if visualize else '(skipped)'}")
        print(f"✓ Embeddings saved to: {emb_path}")

        print("\n" + "="*80)
        print("✓ ANALYSIS COMPLETE!")
        print("="*80)

        return papers, embeddings, clustering_results, best_result

# Example usage
if __name__ == "__main__":
    # Configuration
    API_KEY = os.getenv("OPENAI_API_KEY")
    if not API_KEY:
        print("⚠️ Warning: OPENAI_API_KEY is not set. Please export it before running.")
        print("Example: export OPENAI_API_KEY='your_api_key_here'")

    # Use the uploaded data file
    INPUT_FILE = "arxiv_references_all.txt"
    OUTPUT_FILE = "clustering_results.txt"

    # Initialize clusterer
    clusterer = PaperClusterer(api_key=API_KEY)

    # Run full pipeline
    papers, embeddings, results, best = clusterer.run_full_analysis(
        input_file=INPUT_FILE,
        output_file=OUTPUT_FILE,
        k_values=[3, 4, 5, 6, 7, 8],
        visualize=True
    )

    print(f"\n📊 Best clustering: {best['k']} clusters")
    print(f"📈 Silhouette score: {best['silhouette']:.4f}")
    print(f"\n💾 Results saved to: {OUTPUT_FILE}")
    print(f"🎨 Visualization saved to: clusters_visualization.png")