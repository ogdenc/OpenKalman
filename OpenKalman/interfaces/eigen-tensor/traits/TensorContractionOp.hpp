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
  struct IndexibleObjectTraits<Eigen::TensorContractionOp<Indices, LhsXprType, RhsXprType, OutputKernelType>>
    : Eigen3::IndexibleObjectTraitsBase<Eigen::TensorContractionOp<Indices, LhsXprType, RhsXprType, OutputKernelType>>
  {
  private:

    using T = Eigen::TensorContractionOp<Indices, LhsXprType, RhsXprType, OutputKernelType>;

  public:

    using index_type = typename T::Index;

    template<typename Arg, typename N>
    static constexpr auto get_index_descriptor(const Arg& arg, N n)
    {
      return Eigen::TensorEvaluator<const Arg, Eigen::DefaultDevice>{arg, Eigen::DefaultDevice{}}.dimensions()[static_cast<index_type>(n)];
    }


    // is_one_by_one not defined

    // is_square not defined

    static constexpr bool has_runtime_parameters = true;


    using type = std::tuple<typename LhsXprType::Nested, typename RhsXprType::Nested>;


    template<std::size_t i, typename Arg>
    static decltype(auto) get_nested_matrix(Arg&& arg)
    {
      if constexpr (i == 0)
        return std::forward<Arg>(arg).lhsExpression();
      else
        return std::forward<Arg>(arg).rhsExpression();
      static_assert(i <= 1);
    }


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
        return make_dense_writable_matrix_from(std::forward<Arg>(arg));
      }
    }


    template<typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      using Scalar = scalar_type_of_t<Arg>;

      if constexpr (zero_matrix<LhsXprType>)
      {
        return constant_coefficient{arg.lhsExpression()};
      }
      else if constexpr (zero_matrix<RhsXprType>)
      {
        return constant_coefficient{arg.rhsExpression()};
      }
      else if constexpr (constant_diagonal_matrix<LhsXprType, CompileTimeStatus::any, Likelihood::maybe> and
        constant_matrix<RhsXprType, CompileTimeStatus::any, Likelihood::maybe>)
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
      else if constexpr (constant_matrix<LhsXprType, CompileTimeStatus::any, Likelihood::maybe> and
        constant_diagonal_matrix<RhsXprType, CompileTimeStatus::any, Likelihood::maybe>)
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


    //template<TriangleType t, Likelihood b>
    //static constexpr bool is_triangular = std::tuple_size_v<decltype(std::declval<T>().indices())> == 1 and
    //  triangular_matrix<LhsXprType, t, b> and triangular_matrix<RhsXprType, t, b>;


    static constexpr bool is_triangular_adapter = false;


    static constexpr bool is_hermitian = std::tuple_size_v<decltype(std::declval<T>().indices())> == 1 and
      ((constant_diagonal_matrix<LhsXprType, CompileTimeStatus::any, Likelihood::maybe> and hermitian_matrix<RhsXprType, Likelihood::maybe>) or
      (constant_diagonal_matrix<RhsXprType, CompileTimeStatus::any, Likelihood::maybe> and hermitian_matrix<LhsXprType, Likelihood::maybe>));


    static constexpr bool is_writable = false;


    // data() not defined


    // layout not defined

  };

} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN_TRAITS_TENSORCONTRACTIONOP_HPP
