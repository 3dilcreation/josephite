import { Service } from '../types';

export const services: Service[] = [
  {
    id: 's1',
    name: '3D Printing',
    description: 'High-quality FDM & resin 3D printing with precision up to 0.1mm layer height.',
    icon: '🖨️',
    startingPrice: 199,
    category: 'printing',
    turnaround: '24-48 hours',
    features: [
      'FDM & SLA/Resin printing',
      'Layer height up to 0.1mm',
      'Multiple material options',
      'Color printing available',
      'Volume discounts',
      'Post-processing included',
    ],
    longDescription:
      'Our state-of-the-art 3D printing service offers both FDM (Fused Deposition Modeling) and SLA (Stereolithography) printing technologies. Whether you need functional prototypes, artistic models, or production parts, we deliver precision and quality every time. We support PLA, ABS, PETG, TPU, and premium resin materials.',
    process: [
      'Upload your STL/OBJ file or share your design idea',
      'Our team reviews and optimizes the file for printing',
      'Material and color selection confirmation',
      'Printing begins with quality monitoring',
      'Post-processing (sanding, painting if required)',
      'Quality inspection and delivery',
    ],
    faqs: [
      {
        question: 'What file formats do you accept?',
        answer:
          'We accept STL, OBJ, STEP, IGES, and 3MF files. We can also work with 2D references to create 3D models.',
      },
      {
        question: 'What is the maximum print size?',
        answer:
          'Our FDM printers can print up to 300x300x400mm. For larger objects, we print in parts and assemble.',
      },
      {
        question: 'How do I get a quote?',
        answer:
          'Use our Quote Calculator or send your file to us via WhatsApp or email for an instant quote.',
      },
    ],
  },
  {
    id: 's2',
    name: 'Custom Medals & Trophies',
    description:
      'Premium custom medals and trophies for sports events, corporate awards, and recognition.',
    icon: '🏆',
    startingPrice: 299,
    category: 'awards',
    turnaround: '3-5 days',
    features: [
      'Custom text & logo engraving',
      'Multiple finishes (Gold, Silver, Bronze)',
      'Bulk order discounts',
      'Certificate printing option',
      'Premium packaging',
      'Express delivery available',
    ],
    longDescription:
      'Create lasting memories with our custom-designed medals and trophies. Perfect for sports tournaments, corporate recognition events, school competitions, and special occasions. Each piece is crafted with precision and attention to detail, featuring customizable text, logos, and finishes.',
    process: [
      'Share your design requirements and logo',
      'Our designers create a 3D mockup for approval',
      'Confirm design, quantity, and finish',
      'Production begins with premium materials',
      'Quality check and finishing',
      'Secure packaging and delivery',
    ],
    faqs: [
      {
        question: 'What is the minimum order quantity?',
        answer:
          'We accept orders from 1 piece. Bulk discounts apply for orders of 10+, 50+, and 100+ pieces.',
      },
      {
        question: 'Can you add our company logo?',
        answer:
          'Absolutely! We can engrave or print any logo. Please provide it in high resolution (300 DPI or vector format).',
      },
      {
        question: 'What finishes are available?',
        answer:
          'We offer Gold, Silver, Bronze metallic finishes, as well as custom painted options and natural PLA/ABS colors.',
      },
    ],
  },
  {
    id: 's3',
    name: '3D Modeling & Design',
    description:
      'Professional 3D modeling service to turn your concepts and 2D drawings into print-ready 3D files.',
    icon: '✏️',
    startingPrice: 499,
    category: 'design',
    turnaround: '2-7 days',
    features: [
      'Concept to 3D conversion',
      '2D drawing to 3D model',
      'Reverse engineering',
      'Unlimited revisions (up to 3)',
      'Print-ready file delivery',
      'Multiple export formats',
    ],
    longDescription:
      'Our experienced 3D designers can transform your ideas, sketches, or 2D drawings into professional print-ready 3D models. We specialize in product design, character modeling, architectural visualization, and mechanical parts. All files are optimized for 3D printing with proper wall thickness and support structures.',
    process: [
      'Share your concept via reference images, sketches, or detailed description',
      'Initial consultation to understand requirements',
      'First draft 3D model shared for review',
      'Revisions based on your feedback',
      'Final model optimization for printing',
      'File delivery in STL/OBJ/STEP formats',
    ],
    faqs: [
      {
        question: 'How many revisions are included?',
        answer:
          'Each project includes up to 3 revision rounds. Additional revisions are available at nominal cost.',
      },
      {
        question: 'Do you do character/figurine modeling?',
        answer:
          'Yes! We create custom character models, portraits, anime figures, and miniatures from reference images.',
      },
      {
        question: 'Can you model from a photo?',
        answer:
          'Yes, we can create 3D models from photos using photogrammetry and manual modeling techniques.',
      },
    ],
  },
  {
    id: 's4',
    name: 'Architectural Models',
    description:
      'Detailed scale architectural models for real estate, urban planning, and exhibition purposes.',
    icon: '🏛️',
    startingPrice: 1999,
    category: 'architecture',
    turnaround: '7-14 days',
    features: [
      'Highly detailed scale models',
      'Multiple scales (1:50 to 1:500)',
      'LED lighting option',
      'Landscape and greenery add-ons',
      'Display case included',
      'Client site visit available',
    ],
    longDescription:
      'Our architectural model service caters to architects, real estate developers, urban planners, and interior designers. We create stunning scale models that bring projects to life before construction begins. Models can include LED lighting, landscaping, water features, and surrounding context for presentations and exhibitions.',
    process: [
      'Receive architectural drawings (AutoCAD/PDF)',
      'Scale and complexity assessment',
      'Material and finish selection',
      'Digital 3D model creation from drawings',
      'Staged printing and assembly',
      'Painting, detailing, and LED installation',
      'Quality inspection and delivery',
    ],
    faqs: [
      {
        question: 'What scales do you work with?',
        answer:
          'We work with scales from 1:50 to 1:500. Most popular are 1:100 and 1:200 for presentations.',
      },
      {
        question: 'Can you add LED lighting?',
        answer:
          'Yes! We can add fiber optic or LED strip lighting to illuminate interiors, streets, and features.',
      },
      {
        question: 'Do you handle large projects?',
        answer:
          'Yes, we have handled models for townships, malls, and large residential complexes. Contact us for project-specific quotes.',
      },
    ],
  },
  {
    id: 's5',
    name: 'Statues & Miniatures',
    description:
      'Custom statues, figurines, and miniatures of gods, personalities, and characters — personalized to your specifications.',
    icon: '🗿',
    startingPrice: 799,
    category: 'statues',
    turnaround: '5-10 days',
    features: [
      'Photo-realistic portrait statues',
      'Religious idols and deities',
      'Game/anime character figurines',
      'Historical figure replicas',
      'Hand-painted option',
      'Custom base and display stand',
    ],
    longDescription:
      'Create timeless keepsakes with our custom statue and miniature service. From personalized portrait figurines to religious idols, game characters to historical figures, we bring any design to life with stunning detail. Our artists combine 3D scanning, sculpting, and printing techniques to achieve lifelike results.',
    process: [
      'Share reference photos/design (minimum 3 angles for portraits)',
      'Design consultation and approval',
      '3D sculpting and detailing',
      'Client approval before printing',
      'High-resolution printing',
      'Hand finishing, painting, and base creation',
      'Protective packaging and delivery',
    ],
    faqs: [
      {
        question: 'Can you make a miniature of me?',
        answer:
          'Yes! Share 3-5 photos from different angles and we will create a stunning 3D miniature of you.',
      },
      {
        question: 'What sizes are available?',
        answer:
          'We create miniatures from 5cm to full 30cm statues. Larger sizes available on request.',
      },
      {
        question: 'Are religious idols customizable?',
        answer:
          'Yes, we create Ganesha, Lakshmi, Durga, and other deities in various styles and sizes with customizations.',
      },
    ],
  },
  {
    id: 's6',
    name: 'Industrial Prototypes',
    description:
      'Rapid prototyping for engineering parts, product mockups, and functional prototypes for businesses.',
    icon: '⚙️',
    startingPrice: 999,
    category: 'industrial',
    turnaround: '2-5 days',
    features: [
      'Engineering-grade materials',
      'Tight tolerances (±0.2mm)',
      'Functional testing ready',
      'NDA and confidentiality',
      'Multiple iterations support',
      'Technical consultation included',
    ],
    longDescription:
      'Accelerate your product development with our rapid prototyping service. We work with startups, product designers, and manufacturing companies to create accurate functional prototypes. Our engineering expertise ensures your parts are optimized for printing while maintaining design intent.',
    process: [
      'NDA signing (if required)',
      'CAD file review and DFM analysis',
      'Material selection for function and aesthetics',
      'Prototype printing with tight tolerances',
      'Post-processing and finishing',
      'Dimensional inspection report',
      'Delivery with technical notes',
    ],
    faqs: [
      {
        question: 'What tolerances can you achieve?',
        answer:
          'Standard tolerance is ±0.3mm. With fine settings and resin printing, we achieve ±0.1mm.',
      },
      {
        question: 'Do you sign NDAs?',
        answer:
          'Yes, we sign NDAs for all industrial and commercial projects. Confidentiality is guaranteed.',
      },
      {
        question: 'Can you do metal printing?',
        answer:
          'We offer metal-filled PLA/PETG filaments that look and feel like metal. For actual metal parts, we partner with certified vendors.',
      },
    ],
  },
];
