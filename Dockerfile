FROM nvcr.io/nvidia/pytorch:25.01-py3

WORKDIR /workspace

# Install project dependencies
# (torch, rapids, cudf, flash_attn are already in the base image)
COPY requirements_ngc25.txt .
RUN pip install --no-cache-dir -r requirements_ngc25.txt

# Market scheduler entrypoint — starts daemon in background, then runs CMD
COPY start_services.sh /start_services.sh
RUN chmod +x /start_services.sh

ENTRYPOINT ["/start_services.sh"]

# Default to bash for interactive use; scheduler daemon runs alongside it
CMD ["/bin/bash"]
