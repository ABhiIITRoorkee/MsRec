# MsRec
Multi-Scale Hyper Graph Neural Network-based approach for TPL recommendation



# MsRec for Third-Party Library Recommendation

## Overview

MsRec is a novel Multi-Scale Hyper Graph Neural Network recommending third-party libraries (TPLs) for mobile app development. The model captures low-order and high-order interactions between mobile apps and TPLs by constructing and learning from hypergraphs at multiple scales. This approach enhances the accuracy and diversity of TPL recommendations, making it a robust tool for developers.

## Features

- **Multi-Scale Interaction Modeling**: Captures interactions at the node, hyperedge, and group levels.
- **High-Order Relationship Capture**: Effectively leverages high-order neighbourhood information to understand deeper connections within the app-library graph.
- **Robust Embedding Learning**: Uses Graph Convolutional Networks (GCNs) inspired aggregation to enhance node representations.
- **Comprehensive Recommendation Framework**: Provides accurate and diverse TPL recommendations by considering both the strength and category of interactions within a group.
- **Gaussian Mixture Models (GMM) for Clustering**: Enhances the representation of group-wise interactions.
- **Hinge Loss for Group-wise Interaction**: Improves the model's learning capability.

## Getting Started

### Prerequisites

To run MS-HGNN, you need to have the following installed:

- Python 3.7+
- PyTorch 1.3.1+
- NumPy 1.18.1+
- SciPy 1.3.2+
- Scikit-learn 0.21.3+

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ABhiIITRoorkee/MS-HGNN.git
