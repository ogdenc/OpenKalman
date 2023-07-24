/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Type traits as applied to Eigen::Product.
 */

#ifndef OPENKALMAN_EIGEN3_TRAITS_PRODUCT_HPP
#define OPENKALMAN_EIGEN3_TRAITS_PRODUCT_HPP

#include <type_traits>


namespace OpenKalman::interface
{
  template<typename LhsType, typename RhsType, int Option>
  struct IndexibleObjectTraits<Eigen::Product<LhsType, RhsType, Option>>
    : Eigen3::IndexibleObjectTraitsBase<Eigen::Product<LhsType, RhsType, Option>>
  {
  private:

    using T = Eigen::Product<LhsType, RhsType>;

  public:

    static constexpr std::size_t max_indices = 2;

    template<std::size_t N, typename Arg>
    static constexpr auto get_index_descriptor(const Arg& arg)
    {
      using Xpr = Eigen::Product<LhsType, RhsType>;
      constexpr Eigen::Index dim = N == 0 ? Xpr::RowsAtCompileTime : Xpr::ColsAtCompileTime;

      if constexpr (dim == Eigen::Dynamic)
      {
        if constexpr (N == 0) return static_cast<std::size_t>(arg.rows());
        else return static_cast<std::size_t>(arg.cols());
      }
      else return Dimensions<dim>{};
    }

    static constexpr bool has_runtime_parameters = false;
    using type = std::tuple<typename T::LhsNested, typename T::RhsNested >;

    template<std::size_t i, typename Arg>
    static decltype(auto) get_nested_matrix(Arg&& arg)
    {
      static_assert(i < 2);
      if constexpr (i == 0)
        return std::forward<Arg>(arg).lhs();
      else
        return std::forward<Arg>(arg).rhs();
    }

    template<typename Arg>
    static auto convert_to_self_contained(Arg&& arg)
    {
      using N = Eigen::Product<equivalent_self_contained_t<LhsType>, equivalent_self_contained_t<RhsType>, Option>;
      constexpr Eigen::Index to_be_evaluated_size = self_contained<LhsType> ?
        RhsType::RowsAtCompileTime * RhsType::ColsAtCompileTime :
        LhsType::RowsAtCompileTime * LhsType::ColsAtCompileTime;

      // Do a partial evaluation if at least one argument is self-contained and result size > non-self-contained size.
      if constexpr ((self_contained<LhsType> or self_contained<RhsType>) and
        (LhsType::RowsAtCompileTime != Eigen::Dynamic) and
        (LhsType::ColsAtCompileTime != Eigen::Dynamic) and
        (RhsType::RowsAtCompileTime != Eigen::Dynamic) and
        (RhsType::ColsAtCompileTime != Eigen::Dynamic) and
        ((Eigen::Index)LhsType::RowsAtCompileTime * (Eigen::Index)RhsType::ColsAtCompileTime > to_be_evaluated_size) and
        not std::is_lvalue_reference_v<typename N::LhsNested> and
        not std::is_lvalue_reference_v<typename N::RhsNested>)
      {
        return N {make_self_contained(arg.lhs()), make_self_contained(arg.rhs())};
      }
      else
      {
        return make_dense_writable_matrix_from(std::forward<Arg>(arg));
      }
    }


    template<typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      if constexpr (constant_diagonal_matrix<LhsType, CompileTimeStatus::any, Likelihood::maybe> and
        constant_matrix<RhsType, CompileTimeStatus::any, Likelihood::maybe>)
      {
        return internal::scalar_constant_operation {std::multiplies<>{},
           constant_diagonal_coefficient{arg.lhs()}, constant_coefficient{arg.rhs()}};
      }
      else if constexpr (constant_matrix<LhsType, CompileTimeStatus::any, Likelihood::maybe> and
        constant_diagonal_matrix<RhsType, CompileTimeStatus::any, Likelihood::maybe>)
      {
        return internal::scalar_constant_operation {std::multiplies<>{},
          constant_coefficient{arg.lhs()}, constant_diagonal_coefficient{arg.rhs()}};
      }
      else
      {
        struct Op
        {
          constexpr auto operator()(std::size_t dim, scalar_type_of_t<LhsType> arg1, scalar_type_of_t<RhsType> arg2) const noexcept
          {
            return dim * arg1 * arg2;
          }
        };

        constexpr auto dim = dynamic_dimension<LhsType, 1> ? index_dimension_of_v<RhsType, 0> : index_dimension_of_v<LhsType, 1>;

        if constexpr (zero_matrix<LhsType>) return constant_coefficient{arg.lhs()};
        else if constexpr (zero_matrix<RhsType>) return constant_coefficient{arg.rhs()};
        else if constexpr (constant_diagonal_matrix<LhsType, CompileTimeStatus::any, Likelihood::maybe>)
          return internal::scalar_constant_operation {std::multiplies<>{},
            constant_diagonal_coefficient{arg.lhs()},
            constant_coefficient{arg.rhs()}};
        else if constexpr (constant_diagonal_matrix<RhsType, CompileTimeStatus::any, Likelihood::maybe>)
          return internal::scalar_constant_operation {std::multiplies<>{},
            constant_coefficient{arg.lhs()},
            constant_diagonal_coefficient{arg.rhs()}};
        else if constexpr (dim == dynamic_size)
          return internal::scalar_constant_operation {Op{},
            get_index_dimension_of<1>(arg.lhs()),
            constant_coefficient{arg.rhs()},
            constant_coefficient{arg.lhs()}};
        else
          return internal::scalar_constant_operation {Op{},
            std::integral_constant<std::size_t, dim>{},
            constant_coefficient{arg.rhs()},
            constant_coefficient{arg.lhs()}};
      }
    }


    template<typename Arg>
    static constexpr auto get_constant_diagonal(const Arg& arg)
    {
      return internal::scalar_constant_operation {std::multiplies<>{},
        constant_diagonal_coefficient{arg.lhs()}, constant_diagonal_coefficient{arg.rhs()}};
    }

    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = triangular_matrix<LhsType, t, b> and triangular_matrix<RhsType, t, b>;

    static constexpr bool is_triangular_adapter = false;

    /// A constant diagonal matrix times a hermitian matrix (or vice versa) is hermitian.
    static constexpr bool is_hermitian =
      (constant_diagonal_matrix<LhsType, CompileTimeStatus::any, Likelihood::maybe> and hermitian_matrix<RhsType, Likelihood::maybe>) or
      (constant_diagonal_matrix<RhsType, CompileTimeStatus::any, Likelihood::maybe> and hermitian_matrix<LhsType, Likelihood::maybe>);
  };


} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN3_TRAITS_PRODUCT_HPP
