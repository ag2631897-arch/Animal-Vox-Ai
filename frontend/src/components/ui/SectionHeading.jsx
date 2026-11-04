export default function SectionHeading({ eyebrow, title, description, action, className = '' }) {
  return <div className={`section-heading ${className}`}>
    <div>{eyebrow && <span className="eyebrow">{eyebrow}</span>}<h2>{title}</h2>{description && <p>{description}</p>}</div>
    {action && <div className="section-heading-action">{action}</div>}
  </div>
}
