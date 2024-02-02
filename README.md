## License
This template is licensed under Apache 2.0 and contains the following components: 
* Meta's Llama2 [License](https://ai.meta.com/llama/license/)
* Langchain [MIT](https://github.com/langchain-ai/langchain/blob/master/LICENSE)
* Qdrant [Apace 2.0](https://github.com/qdrant/qdrant/blob/master/LICENSE)

# Llama2 custom Q&A Reference Project

This reference project shows how to use Meta's Llama2 LLM to do Q&A over information that the Llama2 model has not been trained on and will not be able to provide answers out of the box. The project has the following files 

* Llama_Qdrant_RAG.ipynb : This file loads a PDF, converts it to embeddings, stores the embeddings in a local Qdrant Vector Store, defines a prompt, downloads and caches the Llama2 model then constructs a RetrievalQA chain and calls the model to get a response.

* model.py : This file is used to deploy our model as a Domino Model API so we can call it programatically from our application. You must run the Llama_Qdrant_RAG.ipynb to initialise the Qdrant vector store first. It has a `generate` function that should be used as the Model API function. Follow the [instructions in our documentation](https://docs.dominodatalab.com/en/latest/user_guide/8dbc91/deploy-models-at-rest/) to deploy this.

* app.sh : The shell script needed to run the chat app

* API_streamlit_app.py : Streamlit app code for the Q&A chatbot. This app requires the model to be deployed as a Domino Model API and the url / token updating to reference it.

* sample_data/MLOps_whitepaper.pdf : A Domino MLOps whitepaper report that can be used as an example for the flow that has been described above.

* images/domino_banner.png and images/domino_logo.png : Images used in the application.


## Setup instructions

This project requires the following [compute environments](https://docs.dominodatalab.com/en/latest/user_guide/f51038/environments/) to be present. Please ensure the "Automatically make compatible with Domino" checkbox is selected while creating the environment.

You must set your [Workspace volume size to 20GB](https://docs.dominodatalab.com/en/latest/user_guide/0ea71e/change-the-workspace-volume-size/) before running the code to ensure that there is enough space to store the model.


### Environment Requirements

`quay.io/domino/pre-release-environments:project-hub-gpu.main.latest`

Add the following to your dockerfile instructions:
`RUN pip install qdrant_client streamlit_chat pypdf`

**Pluggable Workspace Tools** 
```
jupyterlab:
  title: "JupyterLab"
  iconUrl: "/assets/images/workspace-logos/jupyterlab.svg"
  start: [ "/opt/domino/workspaces/jupyterlab/start" ]
  httpProxy:
    internalPath: "/{{ownerUsername}}/{{projectName}}/{{sessionPathComponent}}/{{runId}}/{{#if pathToOpen}}tree/{{pathToOpen}}{{/if}}"
    port: 8888
    rewrite: false
    requireSubdomain: false
vscode:
 title: "vscode"
 iconUrl: "/assets/images/workspace-logos/vscode.svg"
 start: [ "/opt/domino/workspaces/vscode/start" ]
 httpProxy:
    port: 8888
    requireSubdomain: false
```

Please change the value in `start` according to your Domino version.

### Hardware Requirements
Use the GPU k8s hardware tier for the Workspace and the Model API. The App can be deployed using a Small k8s hardware tier.
