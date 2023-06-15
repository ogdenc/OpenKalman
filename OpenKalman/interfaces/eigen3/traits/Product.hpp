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
#ifndef __cpp_concepts
  template<typename LhsType, typename RhsType, int Option>
  struct IndexTraits<Eigen::Product<LhsType, RhsType, Option>>
    : detail::IndexTraits_Eigen_default<Eigen::Product<LhsType, RhsType, Option>> {};
#endif


  template<typename LhsType, typename RhsType, int Option>
  struct Dependencies<Eigen::Product<LhsType, RhsType, Option>>
  {
  private:

    using T = Eigen::Product<LhsType, RhsType>;

  public:

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
  };


  template<typename Arg1, typename Arg2>
  struct SingleConstant<Eigen::Product<Arg1, Arg2>>
  {
    const Eigen::Product<Arg1, Arg2>& xpr;

    constexpr auto get_constant()
    {
      if constexpr (constant_diagonal_matrix<Arg1, CompileTimeStatus::any, Likelihood::maybe> and
        constant_matrix<Arg2, CompileTimeStatus::any, Likelihood::maybe>)
      {
        return scalar_constant_operation {std::multiplies<>{},
           constant_diagonal_coefficient{xpr.lhs()}, constant_coefficient{xpr.rhs()}};
      }
      else if constexpr (constant_matrix<Arg1, CompileTimeStatus::any, Likelihood::maybe> and
        constant_diagonal_matrix<Arg2, CompileTimeStatus::any, Likelihood::maybe>)
      {
        return scalar_constant_operation {std::multiplies<>{},
          constant_coefficient{xpr.lhs()}, constant_diagonal_coefficient{xpr.rhs()}};
      }
      else
      {
        struct Op
        {
          constexpr auto operator()(std::size_t dim, scalar_type_of_t<Arg1> arg1, scalar_type_of_t<Arg2> arg2) const noexcept
          {
            return dim * arg1 * arg2;
          }
        };

        constexpr auto dim = dynamic_dimension<Arg1, 1> ? index_dimension_of_v<Arg2, 0> : index_dimension_of_v<Arg1, 1>;

        if constexpr (zero_matrix<Arg1>) return constant_coefficient{xpr.lhs()};
        else if constexpr (zero_matrix<Arg2>) return constant_coefficient{xpr.rhs()};
        else if constexpr (constant_diagonal_matrix<Arg1, CompileTimeStatus::any, Likelihood::maybe>)
          return scalar_constant_operation {std::multiplies<>{},
            constant_diagonal_coefficient{xpr.lhs()},
            constant_coefficient{xpr.rhs()}};
        else if constexpr (constant_diagonal_matrix<Arg2, CompileTimeStatus::any, Likelihood::maybe>)
          return scalar_constant_operation {std::multiplies<>{},
            constant_coefficient{xpr.lhs()},
            constant_diagonal_coefficient{xpr.rhs()}};
        else if constexpr (dim == dynamic_size)
          return scalar_constant_operation {Op{},
            get_index_dimension_of<1>(xpr.lhs()),
            constant_coefficient{xpr.rhs()},
            constant_coefficient{xpr.lhs()}};
        else
          return scalar_constant_operation {Op{},
            std::integral_constant<std::size_t, dim>{},
            constant_coefficient{xpr.rhs()},
            constant_coefficient{xpr.lhs()}};
      }
    }

    constexpr auto get_constant_diagonal()
    {
      return scalar_constant_operation {std::multiplies<>{},
        constant_diagonal_coefficient{xpr.lhs()}, constant_diagonal_coefficient{xpr.rhs()}};
    }
  };


  template<typename Arg1, typename Arg2>
  struct TriangularTraits<Eigen::Product<Arg1, Arg2>>
  {
    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = triangular_matrix<Arg1, t, b> and triangular_matrix<Arg2, t, b>;

    static constexpr bool is_triangular_adapter = false;
  };


  /// A constant diagonal matrix times a self-adjoint matrix (or vice versa) is self-adjoint.
#ifdef __cpp_concepts
  template<constant_diagonal_matrix<CompileTimeStatus::any, Likelihood::maybe> Arg1, hermitian_matrix<Likelihood::maybe> Arg2>
  struct HermitianTraits<Eigen::Product<Arg1, Arg2>>
#else
  template<typename Arg1, typename Arg2>
  struct HermitianTraits<Eigen::Product<Arg1, Arg2>, std::enable_if_t<
    (constant_diagonal_matrix<Arg1, CompileTimeStatus::any, Likelihood::maybe> and hermitian_matrix<Arg2, Likelihood::maybe>)>>
#endif
  {
    static constexpr bool is_hermitian = true;
  };


  /// A self-adjoint matrix times a constant-diagonal matrix is self-adjoint.
#ifdef __cpp_concepts
  template<hermitian_matrix<Likelihood::maybe> Arg1, constant_diagonal_matrix<CompileTimeStatus::any, Likelihood::maybe> Arg2>
    requires (not constant_diagonal_matrix<Arg1, CompileTimeStatus::any, Likelihood::maybe>)
  struct HermitianTraits<Eigen::Product<Arg1, Arg2>>
#else
  template<typename Arg1, typename Arg2>
  struct HermitianTraits<Eigen::Product<Arg1, Arg2>, std::enable_if_t<
    (hermitian_matrix<Arg1, Likelihood::maybe> and constant_diagonal_matrix<Arg2, CompileTimeStatus::any, Likelihood::maybe> and
    not constant_diagonal_matrix<Arg1, CompileTimeStatus::any, Likelihood::maybe>)>>
#endif
  {
    static constexpr bool is_hermitian = true;
  };


} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN3_TRAITS_PRODUCT_HPP
