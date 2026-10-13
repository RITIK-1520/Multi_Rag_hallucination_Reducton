"""
Generate a clean, high-resolution architecture diagram for Multi-Agent RAG.
Saves to docs/architecture.png
"""
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_diagram():
    fig, ax = plt.subplots(figsize=(14, 18), dpi=300)
    fig.patch.set_facecolor('#0f172a') # slate-900
    ax.set_facecolor('#0f172a')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 120)
    ax.axis('off')

    # Title & Subtitle
    ax.text(50, 115, "Multi-Agent RAG for Hallucination Reduction", 
            ha='center', va='center', fontsize=22, fontweight='bold', color='#f8fafc')
    ax.text(50, 111.5, "Evidence-First Verification & Self-Correction Pipeline Architecture", 
            ha='center', va='center', fontsize=13, color='#94a3b8', style='italic')

    # Color definitions
    c_user = '#3b82f6'      # blue
    c_analyzer = '#6366f1'  # indigo
    c_retriever = '#0ea5e9' # sky
    c_faiss = '#14b8a6'     # teal
    c_verifier = '#eab308'  # amber
    c_generator = '#8b5cf6' # purple
    c_critic = '#f43f5e'    # rose
    c_decision = '#10b981'  # emerald
    c_output = '#06b6d4'    # cyan

    boxes = [
        {"id": "user", "name": "User Query", "role": "Input natural language prompt / question", "col": c_user, "y": 102, "w": 46, "h": 6.5},
        {"id": "analyzer", "name": "1. Query Analyzer Agent", "role": "Extracts intent, keywords, temporal scope & constraints", "col": c_analyzer, "y": 90, "w": 52, "h": 7},
        {"id": "retriever", "name": "2. Retriever Agent", "role": "Computes semantic query embeddings & retrieves Top-K candidates", "col": c_retriever, "y": 78, "w": 52, "h": 7},
        {"id": "faiss", "name": "FAISS Vector Database", "role": "Dense L2 index storing knowledge base document chunks & metadata", "col": c_faiss, "y": 66, "w": 54, "h": 7},
        {"id": "verifier", "name": "3. Evidence Verification Agent", "role": "Filters low-relevance chunks (< threshold) & extracts key facts", "col": c_verifier, "y": 54, "w": 54, "h": 7},
        {"id": "generator", "name": "4. Answer Generator Agent", "role": "Synthesizes grounded answer strictly utilizing verified facts & citations", "col": c_generator, "y": 42, "w": 54, "h": 7},
        {"id": "critic", "name": "5. Hallucination Critic Agent", "role": "Extracts claims & classifies: Supported, Partial, Unsupported, Contradicted", "col": c_critic, "y": 30, "w": 54, "h": 7},
        {"id": "decision", "name": "6. Decision & Orchestrator", "role": "Evaluates risk against threshold; coordinates regeneration or abstention", "col": c_decision, "y": 18, "w": 54, "h": 7},
        {"id": "output", "name": "Verified Response & Citations", "role": "Delivers final grounded answer, claim audits, and confidence scores", "col": c_output, "y": 6, "w": 50, "h": 6.5},
    ]

    # Draw boxes
    for b in boxes:
        x = 50 - b["w"] / 2
        y = b["y"] - b["h"] / 2
        
        # Glow / shadow effect
        rect_glow = patches.FancyBboxPatch((x-0.4, y-0.4), b["w"]+0.8, b["h"]+0.8,
                                          boxstyle="round,pad=0.8,rounding_size=1.2",
                                          facecolor=b["col"], alpha=0.15, edgecolor='none')
        ax.add_patch(rect_glow)

        # Main box
        rect = patches.FancyBboxPatch((x, y), b["w"], b["h"],
                                      boxstyle="round,pad=0.8,rounding_size=1.0",
                                      facecolor='#1e293b', edgecolor=b["col"], linewidth=2)
        ax.add_patch(rect)

        # Header badge
        badge = patches.FancyBboxPatch((x + 1.5, y + b["h"] - 2.8), b["w"] - 3, 2.4,
                                       boxstyle="round,pad=0.2,rounding_size=0.6",
                                       facecolor=b["col"], alpha=0.25, edgecolor=b["col"], linewidth=1)
        ax.add_patch(badge)

        ax.text(50, b["y"] + 1.2, b["name"], ha='center', va='center',
                fontsize=11.5, fontweight='bold', color='#ffffff')
        ax.text(50, b["y"] - 1.5, b["role"], ha='center', va='center',
                fontsize=9, color='#cbd5e1')

    # Draw downward arrows
    for i in range(len(boxes) - 1):
        y_from = boxes[i]["y"] - boxes[i]["h"]/2 - 0.8
        y_to = boxes[i+1]["y"] + boxes[i+1]["h"]/2 + 0.8
        ax.annotate('', xy=(50, y_to), xytext=(50, y_from),
                    arrowprops=dict(facecolor='#38bdf8', edgecolor='#38bdf8',
                                   arrowstyle="->", lw=2.2, mutation_scale=16))

    # Draw REGENERATION FEEDBACK LOOP from Critic / Decision back to Answer Generator
    x_loop = 82
    y_start = 24  # between Critic & Decision
    y_end = 42    # Answer Generator
    
    ax.annotate('', xy=(50 + 27 + 0.8, y_end), xytext=(x_loop, y_end),
                arrowprops=dict(facecolor='#f43f5e', edgecolor='#f43f5e',
                               arrowstyle="->", lw=2.4, mutation_scale=18))
    
    ax.plot([50 + 27 + 0.8, x_loop], [y_start, y_start], color='#f43f5e', lw=2.4, linestyle='--')
    ax.plot([x_loop, x_loop], [y_start, y_end], color='#f43f5e', lw=2.4, linestyle='--')
    ax.plot([x_loop, 50 + 27 + 1.5], [y_end, y_end], color='#f43f5e', lw=2.4, linestyle='--')

    # Regeneration badge / label
    badge_regen = patches.FancyBboxPatch((x_loop - 10, (y_start + y_end)/2 - 3.5), 20, 7,
                                        boxstyle="round,pad=0.5,rounding_size=0.8",
                                        facecolor='#881337', edgecolor='#f43f5e', linewidth=1.5)
    ax.add_patch(badge_regen)
    ax.text(x_loop, (y_start + y_end)/2 + 1.2, "REGENERATION LOOP",
            ha='center', va='center', fontsize=9, fontweight='bold', color='#fda4af')
    ax.text(x_loop, (y_start + y_end)/2 - 1.4, "If Hallucination Risk > Threshold\n(Refined Prompt with Critic Feedback)",
            ha='center', va='center', fontsize=7.5, color='#fecdd3', multialignment='center')

    # Draw ABSTENTION BRANCH (Left side of Decision)
    x_abs = 15
    y_dec = 18
    ax.plot([50 - 27 - 0.8, x_abs], [y_dec, y_dec], color='#eab308', lw=2, linestyle=':')
    ax.plot([x_abs, x_abs], [y_dec, 6], color='#eab308', lw=2, linestyle=':')
    ax.annotate('', xy=(50 - 25 - 0.8, 6), xytext=(x_abs, 6),
                arrowprops=dict(facecolor='#eab308', edgecolor='#eab308',
                               arrowstyle="->", lw=2, mutation_scale=15))
    
    badge_abs = patches.FancyBboxPatch((x_abs - 8, 10.5), 16, 5.5,
                                      boxstyle="round,pad=0.4,rounding_size=0.6",
                                      facecolor='#713f12', edgecolor='#eab308', linewidth=1.2)
    ax.add_patch(badge_abs)
    ax.text(x_abs, 14, "ABSTENTION PATH",
            ha='center', va='center', fontsize=8, fontweight='bold', color='#fef08a')
    ax.text(x_abs, 12, "Insufficient Evidence\n(\"I do not have enough info\")",
            ha='center', va='center', fontsize=7, color='#fde047', multialignment='center')

    os.makedirs('docs', exist_ok=True)
    plt.tight_layout()
    plt.savefig('docs/architecture.png', facecolor=fig.get_facecolor(), edgecolor='none', dpi=300)
    plt.close()
    print("Architecture diagram saved successfully to docs/architecture.png")

if __name__ == '__main__':
    draw_diagram()
