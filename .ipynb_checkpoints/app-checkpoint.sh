# These top lines are standard configuration to run Streamlit applications in Domino
# If you want to publish a different type of application see our docs: https://docs.dominodatalab.com/en/latest/user_guide/71635d/publish-apps/
mkdir ~/.streamlit
echo "[browser]" > ~/.streamlit/config.toml
echo "gatherUsageStats = true" >> ~/.streamlit/config.toml
echo "serverAddress = \"0.0.0.0\"" >> ~/.streamlit/config.toml
echo "serverPort = 8888" >> ~/.streamlit/config.toml
echo "[server]" >> ~/.streamlit/config.toml
echo "port = 8888" >> ~/.streamlit/config.toml
echo "enableCORS = false" >> ~/.streamlit/config.toml
echo "enableXsrfProtection = false" >> ~/.streamlit/config.toml
cat << EOF >> ~/.streamlit/config.toml
[theme]
base="dark"
EOF

# This line tells Domino which application file to actually run
streamlit run API_streamlit_app.py