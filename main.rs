#![feature(const_cstr_methods)]

use noodles::{
    bcf::{self, header::StringMaps},
    vcf::{Header, record::genotypes::sample}
};
use polars::prelude::*;
use std::{
    error::Error,
    fs::File,
    iter::{repeat},
    path::PathBuf,
};
use itertools::Itertools;
use noodles::csi;
use rayon::{prelude::*, current_num_threads};
use clap::{
    Command, 
    arg, 
    value_parser
};
use sync_file::SyncFile;
use tikv_jemallocator::Jemalloc;

#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

fn cli() -> Command {
    Command::new("infochallenge")
            .args(&[
                arg!(
                    -i --in <FILE> "Path to bcf file with index"
                ).required(true)
                 .value_parser(value_parser!(PathBuf)),
            ])
}


fn main() -> Result<(), Box<dyn Error>> {

    let matches = cli().get_matches();
    let in_path = matches.get_one::<PathBuf>("in").expect("error parsing input path");
    let mut index_path = in_path.clone();
    index_path.set_extension("bcf.csi");


    let mut index_r = File::open(index_path).map(csi::Reader::new)?;
    let index = index_r.read_index()?;

    let f = SyncFile::open(in_path)?;

    let num_samples;
    let stringmaps: StringMaps;
    let header: Header;

    {
        let mut bcf_r = bcf::Reader::new(f.clone());

        bcf_r.read_file_format()?;
        let r_header = bcf_r.read_header()?;
        header = r_header.parse()?;
        stringmaps = r_header.parse()?;

        let mut records = bcf_r.records().peekable();

        let first_record = records.peek().unwrap().clone();
        num_samples = first_record.as_ref().unwrap().genotypes().len();
    }

    let sample_names: Vec<&str> = header.sample_names()
                             .iter()
                             .map(|string| string.split("/").last().unwrap().split(".").next().unwrap())
                             .collect();

    let names_top = "\t".to_string() + sample_names.iter()
                                            .join("\t")
                                            .as_str();

    println!("{names_top}");

    let sizes = index.reference_sequences()
                     .par_iter()
                     .map(|refseq| {
                        let size;
                        if let Some(meta) = refseq.metadata() {
                            size = Some(meta.mapped_record_count() as usize);
                        } else {
                            size = None
                        }

                        size

                     });

    let mut df = (0..).map_while(|i|{
        stringmaps.contigs().get_index(i)
                        }).collect::<Vec<&str>>()
                        .into_par_iter()
                        .zip(sizes)
                        .map(|(chrom, size)| {

        let region = format!("{chrom}").parse().expect("failed to parse region");

        let mut bcf_r;
        { 
            
            bcf_r = bcf::Reader::new(f.clone());
            bcf_r.read_file_format().expect("failed to read format");
            let _r_header = bcf_r.read_header().expect("failed to read header");

        };

        let records = bcf_r.query(stringmaps.contigs(), &index, &region)
                           .expect("failed to query index");
        
        let capacity = if let Some(cap) = size {
            cap
        } else {
            100_000_000 as usize
        };
        
        //preallocate vectors to store data.
        let mut samples: Vec<Vec<Option<i8>>> = (0..num_samples).map(|_| {
            Vec::with_capacity(capacity) 
        }).collect();

        records.for_each(|record| {

            //3 header values then genotypes are pairs
            //get only the GT array, drop other data
            let record = record.expect("failed to read record");
            let (len, _) = record.genotypes()
                                .as_ref()[3..]
                                .into_iter()
                                .find_position(|val| **val == 17)
                                .expect("did not find second field in record");
            let gt = &record.genotypes().as_ref()[3..len + 3];
            
            let gt: Box<dyn Iterator<Item = u8>> = if gt.len() == num_samples { //if only one allele for this record
                Box::new(repeat(10).take(num_samples)) //10 is a marker value to filter on later
            } else {
                Box::new(gt.chunks_exact(2)
                    .map(|chunk| { 

                        if chunk.iter().any( |val| (val == &6) | (val == &8) ) { //if a dual/triple snp
                            return 10 
                        } else { //combine good values
                            chunk[0] + chunk[1]
                        }

                    }))
            };
            
            gt.zip(samples.iter_mut())
                .for_each(|(read_value, sample)| {
 
                    //map allels to helpfull numbers otherwise replace with None
                    let value = match read_value { 
                        0 => None,
                        10 => Some(10_i8), //to be used for filtering
                        4 => Some(0_i8),
                        6 => Some(1_i8),
                        8 => Some(2_i8),
                        _ => panic!("{read_value}")
                    };

                    sample.push(value);

           });


        });


        let data = samples.into_par_iter()
                .map(|sample| {
                    let chunked = sample.into_iter().collect();
                    chunked
                })
                .collect::<Vec<ChunkedArray<Int8Type>>>();
        
        data
    }).reduce(
        || (0..num_samples).map(|_| ChunkedArray::<Int8Type>::default())
                            .collect::<Vec<ChunkedArray<Int8Type>>>(),
        |a_s, b_s| {
            a_s.into_iter()
             .zip(b_s)
             .map(|(mut a, b)| {
                a.append(&b);
                a
             }).collect::<Vec<ChunkedArray<Int8Type>>>()
        }
    ).into_iter()
    .zip(sample_names.iter())
    .map(|(mut chunkar, name)| {
        chunkar.rename(name);
        chunkar.into_series()
    }).collect::<DataFrame>();             

    df = {
        let mask = df.iter() //create a mask of all records without dual snps
                .map(|array| (*array).i8().unwrap().not_equal(10))
                .reduce(|accum, array| accum & array)
                .unwrap();

        df = df.filter(&mask).unwrap();
        df.agg_chunks()

        

    };
    //let data = Arc::new(data);

    //preinitiallize matrix with 0.0 
    let mut matrix = (0_usize..num_samples).map(|_| repeat(0.0).take(num_samples).collect())
                                            .collect::<Vec<Vec<f64>>>();

    
    //returns all combinations of all numbers in the sequence as tuples
    sample_names.into_iter().tuple_combinations::<(&str, &str)>()
                        .collect::<Vec<(&str, &str)>>()
                        .par_chunks((1..num_samples).sum::<usize>() / current_num_threads() / 2)
                        .map(|chunk| {
                            chunk.into_iter().map(|(i, j)| {
                                (&df[*i] - &df[*j]).abs()
                                  .unwrap()
                                  .mean()
                                  .unwrap_or(f64::NAN) / 2.0
                                
    
                            }).collect_vec()
                        }).flatten()
                        .collect::<Vec<f64>>()
                        .into_iter()
                        .zip((0_usize..num_samples).tuple_combinations::<(usize, usize)>())
                        .for_each(|(avg, (i, j))| {
                            //store values in the matrix in both directions
                            matrix[j][i] = avg;
                            matrix[i][j] = avg;

                        });

    matrix.iter()
          .zip(header.sample_names().iter())
          .for_each(|(line, samp_name)| {
            let line = line.iter()
                           .join("\t");
            println!("{samp_name}\t{line}");
    });


    Ok(())
}
