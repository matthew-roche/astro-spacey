
const CHANGES = [
    {"release": "version: alpha-0.0.3, date: 2025 October 20", "content":[
        "Fixed UI issues.",
        "Added UI loading indicaor.",
        "Model Deployment"
    ]},
    {"release": "version: alpha-0.0.2, date: 2025 October 17", "content":[
        "Model refinement for abstaining from answering.", 
        "Critical Evaluation of the model."
    ]},
    {"release": "version: alpha-0.0.1, date: 2025 October 7", "content":[
        "Zero-shot AI development", 
        "Initial Testing"
    ]}
]

export default function ChangeLog() {
    return <div className='justify-items-center-safe'>
        <div className='w-full max-w-7xl pt-2 ps-2'>
            <h1 className="text-2xl font-semibold mb-4">Changes. A lot more to come.</h1>
            <ul className="mt-6 space-y-6">
                {CHANGES.map(change => {
                return <li className="rounded-2xl border border-gray-200 bg-white p-6 shadow-sm hover:shadow transition-shadow">
                    <h3 className="text-lg font-semibold text-gray-900">{change.release}</h3>
                    {change.content.map(item => {
                        return <p className="mt-2 text-sm leading-6 text-gray-600">{item}</p>
                    })}
                </li>
                })}
            </ul>
        </div>
    </div>
}