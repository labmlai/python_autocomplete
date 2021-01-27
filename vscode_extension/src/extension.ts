import * as vscode from 'vscode'
import * as https from 'http'

async function fetch(prompt: string) {
	const data = JSON.stringify({ prompt: prompt })

	return new Promise<any>((resolve, reject) => {
		let respone_data = ''
		const req = https.request('http://localhost:5000/autocomplete', {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
				'Content-Length': data.length
			}
		}, (res) => {
			res.on('data', (d) => {
				respone_data += d
			})

			res.on('end', () => {
				resolve(JSON.parse(respone_data))
			})
		})
		req.on('error', (error) => {
			console.error(error)
			reject()
		})
		req.write(data)
		req.end()
	})
}

function getPrompt(document: vscode.TextDocument, position: vscode.Position) {
	const start = Math.max(0, position.line - 20)
	let text = ''
	for (let i = start; i < position.line; ++i) {
		text += document.lineAt(i).text + '\n'
	}
	const line = document.lineAt(position).text
	text += line.substr(0, position.character)

	return text
}

export function activate(context: vscode.ExtensionContext) {
	const provider = vscode.languages.registerCompletionItemProvider('python', {
		async provideCompletionItems(document: vscode.TextDocument, position: vscode.Position, token: vscode.CancellationToken, context: vscode.CompletionContext) {
			const prompt = getPrompt(document, position)
			let response

			try {
				response = await fetch(prompt)
			} catch(e) {
				return []
			}

			// Failure
			if (!response.success) {
				return []
			}

			let prediction: string = response.prediction

			// Remove new lines because it's a bit annoying?
			let nl = prediction.indexOf('\n')
			if (nl !== -1) {
				prediction = prediction.substr(0, nl)
			}

			if (prediction === '') {
				// If at end of a line just predict new line, to avoid annoying default vscode predictions
				if (nl !== -1) {
					const simpleCompletion = new vscode.CompletionItem('\n')
					simpleCompletion.kind = vscode.CompletionItemKind.Text
					simpleCompletion.command = { command: 'editor.action.triggerSuggest', title: 'Re-trigger completions...' }
					return [simpleCompletion]
				}
				return []
			}

			// Add any word prefix from text (because thats how vscode works)
			let range = document.getWordRangeAtPosition(position)
			if (range != null) {
				const line = document.lineAt(position).text
				let prefix = line.substring(range.start.character, position.character)
				prediction = prefix + prediction
			}

			// Create a completion
			const simpleCompletion = new vscode.CompletionItem(prediction)
			simpleCompletion.kind = vscode.CompletionItemKind.Text
			// Dont trigger autocompletion if we hit a new line
			if (nl === -1) {
				simpleCompletion.command = { command: 'editor.action.triggerSuggest', title: 'Re-trigger completions...' }
			}

			return [simpleCompletion]
		}
	})

	context.subscriptions.push(provider)
}
