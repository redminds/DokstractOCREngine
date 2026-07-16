# Testing

Run the checks:

```bash
make validate
```

The test suite covers:

- capability and API-version compatibility
- immutable release registration
- promotion-based reassignment
- rollback to the previous known-good release
- internal and admin API auth
