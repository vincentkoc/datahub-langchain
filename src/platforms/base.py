class BaseIngestor:
    """Base class for all ingestors."""

    def fetch_data(self):
        """Fetch data from the source platform."""
        raise NotImplementedError("fetch_data() must be implemented by subclasses.")

    def process_data(self, raw_data):
        """Process raw data into MCEs."""
        raise NotImplementedError("process_data() must be implemented by subclasses.")

    def emit_data(self, processed_data):
        """Emit or save processed MCEs."""
        raise NotImplementedError("emit_data() must be implemented by subclasses.")
