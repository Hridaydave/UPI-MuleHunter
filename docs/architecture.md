# System Architecture

## High-Level Architecture

```text
                 Transaction Input
                        |
                        v
              Intent / Authority Layer
                        |
          +-------------+-------------+
          |             |             |
          v             v             v
      Behavioral     Temporal       Graph
        Model         Model         Model
         RF          Markov       GraphSAGE
          |             |             |
          +-------------+-------------+
                        |
                        v
                  Risk Fusion
                        |
            +-----------+-----------+
            |                       |
            v                       v
        Mule Risk              Intent Risk
            |                       |
            +-----------+-----------+
                        |
                        v
                 Decision Engine
                        |
             +----------+----------+
             |          |          |
             v          v          v
           ALLOW      REVIEW      HOLD
