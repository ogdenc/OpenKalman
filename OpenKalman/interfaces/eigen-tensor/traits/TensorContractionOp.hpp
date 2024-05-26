/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Type traits as applied to Eigen::TensorContractionOp
 */

#ifndef OPENKALMAN_EIGEN_TRAITS_TENSORCONTRACTIONOP_HPP
#define OPENKALMAN_EIGEN_TRAITS_TENSORCONTRACTIONOP_HPP


namespace OpenKalman::interface
{
  template<typename Indices, typename LhsXprType, typename RhsXprType, typename OutputKernelType>
  struct indexible_object_traits<Eigen::TensorContractionOp<Indices, LhsXprType, RhsXprType, OutputKernelType>>
    : Eigen3::indexible_object_traits_tensor_base<Eigen::TensorContractionOp<Indices, LhsXprType, RhsXprType, OutputKernelType>>
  {
  private:

    using Xpr = Eigen::TensorContractionOp<Indices, LhsXprType, RhsXprType, OutputKernelType>;
    using Base = Eigen3::indexible_object_traits_tensor_base<Xpr>;

  public:

    template<typename Arg, typename N>
    static constexpr std::size_t get_vector_space_descriptor(const Arg& arg, N n)
    {
      using IndexType = typename Xpr::Index;
      return Eigen::TensorEvaluator<const Arg, Eigen::DefaultDevice>{arg, Eigen::DefaultDevice{}}.dimensions()[static_cast<IndexType>(n)];
    }


    using dependents = std::tuple<typename LhsXprType::Nested, typename RhsXprType::Nested>;


    static constexpr bool has_runtime_parameters = true;


    // nested_object() not defined


    template<typename Arg>
    static auto convert_to_self_contained(Arg&& arg)
    {
      using N = Eigen::TensorContractionOp<Indices, equivalent_self_contained_t<LhsXprType>, equivalent_self_contained_t<RhsXprType>, OutputKernelType>;
      // Do a partial evaluation as long as at least one argument is already self-contained.
      if constexpr ((self_contained<LhsXprType> or self_contained<RhsXprType>) and
        not std::is_lvalue_reference_v<typename LhsXprType::Nested> and
        not std::is_lvalue_reference_v<typename RhsXprType::Nested>)
      {
        return N {make_self_contained(arg.lhsExpression()), make_self_contained(arg.rhsExpression()), arg.indices(), arg.outputKernel()};
      }
      else
      {
        return make_dense_object(std::forward<Arg>(arg));
      }
    }


    template<typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      using Scalar = scalar_type_of_t<Arg>;

      if constexpr (zero<LhsXprType>)
      {
        return constant_coefficient{arg.lhsExpression()};
      }
      else if constexpr (zero<RhsXprType>)
      {
        return constant_coefficient{arg.rhsExpression()};
      }
      else if constexpr (constant_diagonal_matrix<LhsXprType> and constant_matrix<RhsXprType>)
      {
        if constexpr (std::tuple_size_v<decltype(arg.indices())> == 1)
        {
          return constant_diagonal_coefficient{arg.lhsExpression()} * constant_coefficient{arg.rhsExpression()};
        }
        else
        {
          auto& indices = arg.indices();
          auto dims = Eigen::TensorEvaluator<const Arg, Eigen::DefaultDevice>{arg, Eigen::DefaultDevice{}}.dimensions();
          auto f = [&dims](const Scalar& a, auto b) -> Scalar { return a * dims[b.first]; };
          auto factor = std::accumulate(++indices.cbegin(), indices.cend(), Scalar{1}, f);
          return factor * (constant_diagonal_coefficient{arg.lhsExpression()} * constant_coefficient{arg.rhsExpression()});
        }
      }
      else if constexpr (constant_matrix<LhsXprType> and constant_diagonal_matrix<RhsXprType>)
      {
        if constexpr (std::tuple_size_v<decltype(arg.indices())> == 1)
        {
          return constant_coefficient{arg.lhsExpression()} * constant_diagonal_coefficient{arg.rhsExpression()};
        }
        else
        {
          auto& indices = arg.indices();
          auto dims = Eigen::TensorEvaluator<const Arg, Eigen::DefaultDevice>{arg, Eigen::DefaultDevice{}}.dimensions();
          auto f = [&dims](const Scalar& a, auto b) -> Scalar { return a * dims[b.first]; };
          auto factor = std::accumulate(++indices.cbegin(), indices.cend(), Scalar{1}, f);
          return factor * (constant_coefficient{arg.lhsExpression()} * constant_diagonal_coefficient{arg.rhsExpression()});
        }
      }
      else
      {
        auto& indices = arg.indices();
        auto dims = Eigen::TensorEvaluator<const Arg, Eigen::DefaultDevice>{arg, Eigen::DefaultDevice{}}.dimensions();
        auto f = [&dims](const Scalar& a, auto b) -> Scalar { return a * dims[b.first]; };
        auto factor = std::accumulate(indices.cbegin(), indices.cend(), Scalar{1}, f);
        return factor * (constant_coefficient{arg.lhsExpression()} * constant_coefficient{arg.rhsExpression()});
      }
    }


    template<typename Arg>
    static constexpr auto get_constant_diagonal(const Arg& arg)
    {
      if constexpr (std::tuple_size_v<decltype(arg.indices())> == 1)
      {
        return internal::scalar_constant_operation {std::multiplies<>{},
          constant_diagonal_coefficient{arg.lhs()}, constant_diagonal_coefficient{arg.rhs()}};
      }
      else
      {
        using Scalar = scalar_type_of_t<Arg>;
        auto& indices = arg.indices();
        auto dims = Eigen::TensorEvaluator<const Arg, Eigen::DefaultDevice>{arg, Eigen::DefaultDevice{}}.dimensions();
        auto f = [&dims](const Scalar& a, auto b) -> Scalar { return a * dims[b.first]; };
        auto factor = std::accumulate(++indices.cbegin(), indices.cend(), Scalar{1}, f);
        return factor * (constant_diagonal_coefficient{arg.lhsExpression()} * constant_coefficient{arg.rhsExpression()});
      }
    }


    // one_dimensional not defined

    // is_square not defined

    //template<TriangleType t>
    //static constexpr bool is_triangular = std::tuple_size_v<decltype(std::declval<T>().indices())> == 1 and
    //  triangular_matrix<LhsXprType, t> and triangular_matrix<RhsXprType, t>;


    static constexpr bool is_triangular_adapter = false;


    static constexpr bool is_hermitian = std::tuple_size_v<decltype(std::declval<Xpr>().indices())> == 1 and
      ((constant_diagonal_matrix<LhsXprType> and hermitian_matrix<RhsXprType, Qualification::depends_on_dynamic_shape>) or
      (constant_diagonal_matrix<RhsXprType> and hermitian_matrix<LhsXprType, Qualification::depends_on_dynamic_shape>));


    static constexpr bool is_writable = false;


    // raw_data() not defined


    // layout not defined

  };

} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN_TRAITS_TENSORCONTRACTIONOP_HPP
