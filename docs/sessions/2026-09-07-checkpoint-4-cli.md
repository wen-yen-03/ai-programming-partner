create command -
>>   --file sessions/demo.json `
>>   --goal "Build a calculator" `
>>   --project-path calculator `
>>   --constraints "Use Python standard library only" `
>>   --acceptance-criteria "All tests pass" `
>>   --next-action "Write the first test"
Goal: Build a calculator
Project Path: calculator
Constraints: Use Python standard library only
Acceptance Criteria: All tests pass
Status: active
Evidence: 
Explanation Checkpoint: 
Next Action: Write the first test

show command -
PS ...\ai-programming-partner> python -m partner.cli show --file sessions/demo.json
Goal: Build a calculator
Project Path: calculator
Constraints: Use Python standard library only
Acceptance Criteria: All tests pass
Status: active
Evidence: 
Explanation Checkpoint: 
Next Action: Write the first test

complete command -
PS ...\ai-programming-partner> python -m partner.cli complete `
>>   --file sessions/demo.json `
>>   --evidence "python -m pytest: 14 passed" `
>>   --explanation "I understand the model, storage, CLI, and completion gate."

show command for demo file -
PS ...\ai-programming-partner> python -m partner.cli show --file sessions/demo.json
Goal: Build a calculator
Project Path: calculator
Constraints: Use Python standard library only
Acceptance Criteria: All tests pass
Status: completed
Evidence: python -m pytest: 14 passed
Explanation Checkpoint: I understand the model, storage, CLI, and completion gate.
Next Action: Write the first test

pytest -
PS ...\ai-programming-partner> python -m pytest
============================================================================================================== test session starts ===============================================================================================================
platform win32 -- Python 3.14.7, pytest-9.1.1, pluggy-1.6.0
rootdir: ...\ai-programming-partner
configfile: pyproject.toml
testpaths: tests
plugins: anyio-4.14.2
collected 14 items                                                                                                                                                                                                                                

tests\test_cli.py .......                                                                                                                                                                                                                   [ 50%]
tests\test_models.py .....                                                                                                                                                                                                                  [ 85%]
tests\test_storage.py ..                                                                                                                                                                                                                    [100%]

=============================================================================================================== 14 passed in 0.07s ===============================================================================================================

when evidence is blank, the comlpete command raised the IncompleteSessionError

CLI calls complete to make sure the session is completed and updated correctly before calling the save command to store the json file

storage merely save/retrieves the json file, it does not interact with modification or decisioning of the files

When evidence is blank, Session.complete() raises IncompleteSessionError.
The CLI catches it, prints the error, and does not save the session as completed.

Status: completed
Evidence: python -m pytest: 14 passed