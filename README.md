# MindAid


## Inspiration


The inspiration for <ins>**MindAid**</ins> comes from our experiences living in small cities, where access to good counselors and mental health support is often limited or nonexistent. Despite the growing awareness and recognition of mental health issues globally, millions of people still face significant barriers to accessing the care they need due to stigma, lack of resources, geographic limitations, and financial constraints.
Moreover, many people hesitate to visit doctors due to fear, further widening the gap between those in need and the support they require. This deeply concerns us, and we believe technology can offer a solution.
MindAid was conceived out of a deep concern for these challenges and a strong belief in the transformative potential of technology. We recognized that artificial intelligence could play a pivotal role in bridging the gap between those in need and the mental health support they require. Our aim is to create a platform that can provide immediate, accessible, and reliable mental health assistance to anyone, anywhere, regardless of their circumstances.
By addressing the shortcomings of traditional mental health care, MindAid aspires to contribute to a world where mental health support is not a privilege, but a readily available resource for all.
Our mission is to empower individuals to take control of their mental health, break down the stigma associated with mental health issues, and promote a culture of understanding and support.


## What it does


MindAid is a comprehensive platform designed to support mental health in various ways:


- **Diagnostic Model**: MindAid employs an advanced diagnostic model capable of identifying whether a user is experiencing one of several mental health issues, including depression, anxiety, PTSD, and addiction.
- **AI-Powered Counselor**: Once a diagnosis is made, MindAid offers an AI-powered counselor to provide tailored counseling for the identified problems. This AI counselor is designed to offer support and guidance for managing and overcoming mental health challenges. It is also able to handle the counselling if the patient is going through more than one of the issues mentioned above.
By combining diagnostic capabilities with AI-driven counseling, MindAid provides a robust and accessible solution for individuals seeking mental health support.


## How we built it


- **Website**: The website is built using Flask, a lightweight web framework for Python. For the database, we have utilized SQLite 3 to store and manage user data securely and efficiently.

- **Diagnostic Model**: We employed the BERT-base-multilingual-cased model from Hugging Face, fine-tuning it to predict various mental health disorders. The dataset used for fine-tuning was sourced from [Zenodo](https://zenodo.org/records/3941387), providing a robust foundation for accurate diagnostics.

- **AI Counselor**: For the AI-powered counselor, we used LLaMA3-70B-8192 via Groq. To enhance the quality of counseling responses, we applied Retrieval-Augmented Generation (RAG) using LangChain, which allows the model to generate more informed and contextually relevant responses.
- **Deployment**: The entire application is deployed on Microsoft Azure, ensuring scalability, reliability, and security for our users.


## Challenges we faced


- **Data Acquisition**: Obtaining relevant and high-quality data for creating the diagnostic bot for mental health disorders was a significant hurdle.
- **Computational Resource Limitations**: Due to a lack of computational resources, we were unable to further enhance the counseling model.
- **Deployment Issues**: Deploying multiple large language models **on Azure** proved to be challenging due to the complexity and resource requirements.


## Accomplishments


Since its inception, MindAid has achieved several significant milestones:


- **Successful Deployment**: The platform is fully functional and deployed on Microsoft Azure, providing a scalable and secure environment for users.
- **Accurate Diagnostics**: Our fine-tuned BERT-based diagnostic model delivers reliable and accurate predictions for mental health disorders, aiding in early detection and intervention.
- **Advanced AI Counseling**: The integration of LLaMA3-70B-8192 with **Retrieval-Augmented Generation** has enabled the AI counselor to offer high-quality, context-aware guidance, supporting users through their mental health journeys.


## What's next for MindAid


- **Adding More Disorders**: We plan to expand our diagnostic capabilities to include a wider range of mental health disorders, providing even more comprehensive support.
- **Increasing Dataset Size**: By incorporating larger and more diverse datasets, we aim to enhance the accuracy and reliability of our diagnostic model.
- **Leveraging Better LLMs**: With adequate funding and access to superior computational resources, we can utilize more advanced language models and apply RAG to improve counseling responses.
- **Remote Doctor Consultations**: We aim to add an interface that allows users to connect remotely with mental health professionals for additional consultation and support.
- **Commercialization**: We plan to commercialize MindAid, making it available to a broader audience and ensuring its sustainability and continuous improvement.
   
## Demonstration video




## Live Deployed link

https://mindaid.azurewebsites.net/
