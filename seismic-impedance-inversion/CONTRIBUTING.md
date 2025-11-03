# Contributing to Seismic Impedance Inversion

Thank you for your interest in contributing to this project!

## How to Contribute

### Reporting Issues

If you find a bug or have a suggestion:
1. Check if the issue already exists
2. Create a detailed issue with:
   - Clear description
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - System information

### Code Contributions

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Test thoroughly
5. Commit with clear messages
6. Push to your fork
7. Create a Pull Request

### Coding Standards

- Follow PEP 8 for Python code
- Add docstrings to all functions and classes
- Include type hints where appropriate
- Write unit tests for new features
- Update documentation

### Testing

Before submitting:
```bash
# Test model
python model.py

# Test data loading
python data_loading.py

# Test utilities
python utils.py

# Run a quick training test
python train.py --model 1d --epochs 2 --num_train_samples 20
```

## Questions?

Feel free to open an issue for any questions or discussions.
