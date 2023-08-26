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
 * \brief Trait details for Eigen binary functors used in PartialReduxExpr.
 */

#ifndef OPENKALMAN_EIGEN_TRAITS_FUNCTORS_REDUX_HPP
#define OPENKALMAN_EIGEN_TRAITS_FUNCTORS_REDUX_HPP

#include <type_traits>

namespace OpenKalman::Eigen3
{
  namespace detail
  {
    template<typename XprType>
    struct is_diag : std::bool_constant<
      zero_matrix<XprType> or one_by_one_matrix<XprType> ? true :
      constant_matrix<XprType> ? false :
      constant_diagonal_matrix<XprType, CompileTimeStatus::any, Likelihood::maybe> ? true :
      constant_matrix<XprType, CompileTimeStatus::any, Likelihood::maybe> ? std::false_type{} : false> {};

    template<typename XprType>
    constexpr bool is_diag_v = is_diag<XprType>::value;
  } // namespace detail


  // Default, if MemberOp is not handled below.
  template<typename XprType, typename MemberOp>
  struct SingleConstantPartialRedux
  {
    template<typename C, typename Dim, typename Factor>
    static constexpr auto get_constant(const C& c, const Dim& dim, const Factor& factor) noexcept
    {
      return std::monostate{};
    }
  };


  /////////////
  //  Norms  //
  /////////////

  template<int p, typename XprType, typename...Args>
  struct SingleConstantPartialRedux<XprType, Eigen::internal::member_lpnorm<p, Args...>>
  {
    struct Op
    {
      template<typename Scalar>
      constexpr Scalar operator()(Scalar x, std::size_t dim) const noexcept
      {
        auto arg = internal::constexpr_abs(x);
        if constexpr (p == 1) return static_cast<Scalar>(dim) * arg;
        else if constexpr (p == 2) return internal::constexpr_sqrt(static_cast<Scalar>(dim)) * arg;
        else if constexpr (p == Eigen::Infinity) return arg;
        else return internal::constexpr_pow(static_cast<Scalar>(dim), static_cast<Scalar>(1)/p) * arg;
      }
    };

    template<typename C, typename Dim, typename Factor>
    static constexpr auto get_constant(const C& c, const Dim& dim, const Factor& factor) noexcept
    {
      using Scalar = std::decay_t<decltype(get_scalar_constant_value(c))>;
      if constexpr (zero_matrix<XprType>) return internal::ScalarConstant<Likelihood::definitely, Scalar, 0>{};
      else if constexpr (p == 0)
      {
        if constexpr (std::numeric_limits<Scalar>::has_infinity) return std::numeric_limits<Scalar>::infinity();
        else throw std::domain_error {"Domain error in lpnorm<0>: result is infinity"};
      }
      else if constexpr (detail::is_diag_v<XprType>) return internal::scalar_constant_operation {Op{}, c, factor};
      else return internal::scalar_constant_operation {Op{}, c, dim * factor};
    }
  };


  template<typename XprType, typename...Args>
  struct SingleConstantPartialRedux<XprType, Eigen::internal::member_stableNorm<Args...>>
  {
    template<typename C, typename Dim, typename Factor>
    static constexpr auto get_constant(const C& c, const Dim& dim, const Factor& factor) noexcept
    {
      using Scalar = std::decay_t<decltype(get_scalar_constant_value(c))>;
      if constexpr (zero_matrix<XprType>) return internal::ScalarConstant<Likelihood::definitely, Scalar, 0>{};
      else if constexpr (detail::is_diag_v<XprType>) return internal::constexpr_sqrt(factor) * internal::constexpr_abs(c);
      else return internal::constexpr_sqrt(dim * factor) * internal::constexpr_abs(c);
    }
  };


  template<typename XprType, typename...Args>
  struct SingleConstantPartialRedux<XprType, Eigen::internal::member_hypotNorm<Args...>>
  {
    template<typename C, typename Dim, typename Factor>
    static constexpr auto get_constant(const C& c, const Dim& dim, const Factor& factor) noexcept
    {
      using Scalar = std::decay_t<decltype(get_scalar_constant_value(c))>;
      if constexpr (zero_matrix<XprType>) return internal::ScalarConstant<Likelihood::definitely, Scalar, 0>{};
      else if constexpr (detail::is_diag_v<XprType>) return internal::constexpr_sqrt(factor) * internal::constexpr_abs(c);
      else return internal::constexpr_sqrt(dim * factor) * internal::constexpr_abs(c);
    }
  };


# if not EIGEN_VERSION_AT_LEAST(3,4,0)
  template<typename XprType, typename...Args>
  struct SingleConstantPartialRedux<Eigen::internal::member_squaredNorm<Args...>>
  {
    struct Op
    {
      template<typename Scalar>
      constexpr Scalar operator()(Scalar x, std::size_t dim) const noexcept
      {
        if constexpr (complex_number<Scalar>)
        {
          auto r = internal::constexpr_real(x);
          auto i = internal::constexpr_imag(x);
          if constexpr (detail::is_diag_v<XprType>) return r * r + i * i;
          else return dim * (r * r + i * i);
        }
        else return dim * x * x;
      }
    };

    template<typename C, typename Dim, typename Factor>
    static constexpr auto get_constant(const C& c, const Dim& dim, const Factor& factor) noexcept
    {
      using Scalar = std::decay_t<decltype(get_scalar_constant_value(c))>;
      if constexpr (zero_matrix<XprType>) return internal::ScalarConstant<Likelihood::definitely, Scalar, 0>{};
      else if constexpr (detail::is_diag_v<XprType>) return scalar_constant_operation {Op{}, c, factor};
      else return scalar_constant_operation {Op{}, c, dim * factor};
    }
  };

  template<typename XprType, typename...Args>
  struct SingleConstantPartialRedux<Eigen::internal::member_norm<Args...>>
  {
    struct Op
    {
      template<typename Scalar>
      constexpr Scalar operator()(Scalar x, std::size_t dim) const noexcept
      {
        if constexpr (complex_number<Scalar>)
        {
          auto r = internal::constexpr_real(x);
          auto i = internal::constexpr_imag(x);
          if constexpr (detail::is_diag_v<XprType>) return r * r + i * i;
          return internal::constexpr_sqrt(dim * (r * r + i * i));
        }
        else
        {
          auto arg = internal::constexpr_abs(x);
          return internal::constexpr_sqrt(static_cast<Scalar>(dim)) * arg;
        }
      }
    };

    template<typename C, typename Dim, typename Factor>
    static constexpr auto get_constant(const C& c, const Dim& dim, const Factor& factor) noexcept
    {
      using Scalar = std::decay_t<decltype(get_scalar_constant_value(c))>;
      if constexpr (zero_matrix<XprType>) return internal::ScalarConstant<Likelihood::definitely, Scalar, 0>{};
      else if constexpr (detail::is_diag_v<XprType>) return scalar_constant_operation {Op{}, c, factor};
      else return scalar_constant_operation {Op{}, c, dim * factor};
    }
  };

  template<typename XprType, typename...Args>
  struct SingleConstantPartialRedux<Eigen::internal::member_mean<Args...>>
  {
    template<typename C, typename Dim, typename Factor>
    static constexpr auto get_constant(const C& c, const Dim& dim, const Factor& factor) noexcept
    {
      using Scalar = std::decay_t<decltype(get_scalar_constant_value(c))>;
      if constexpr (zero_matrix<XprType>) return internal::ScalarConstant<Likelihood::definitely, Scalar, 0>{};
      else if constexpr (not detail::is_diag_v<XprType>) return c
      else return (c * factor) / dim;
    }
  };
# endif


  ///////////
  //  sum  //
  ///////////

  template<typename XprType, typename T, typename S>
  struct SingleConstantPartialRedux<XprType, Eigen::internal::member_redux<std::plus<T>, S>>
  {
    struct Op
    {
      template<typename Scalar>
      constexpr Scalar operator()(Scalar x, std::size_t dim) const noexcept { return x * static_cast<Scalar>(dim); }
    };

    template<typename C, typename Dim, typename Factor>
    static constexpr auto get_constant(const C& c, const Dim& dim, const Factor& factor) noexcept
    {
      using Scalar = std::decay_t<decltype(get_scalar_constant_value(c))>;
      if constexpr (zero_matrix<XprType>) return internal::ScalarConstant<Likelihood::definitely, Scalar, 0>{};
      else if constexpr (detail::is_diag_v<XprType>) return internal::scalar_constant_operation {Op{}, c, factor};
      else return internal::scalar_constant_operation {Op{}, c, dim * factor};
    }
  };


  template<typename XprType, typename...Args>
  struct SingleConstantPartialRedux<XprType, Eigen::internal::member_sum<Args...>>
    : SingleConstantPartialRedux<XprType, Eigen::internal::member_redux<std::plus<scalar_type_of_t<XprType>>, scalar_type_of_t<XprType>>> {};


  template<typename XprType, typename Scalar1, typename Scalar2, typename S>
  struct SingleConstantPartialRedux<XprType, Eigen::internal::member_redux<Eigen::internal::scalar_sum_op<Scalar1, Scalar2>, S>>
    : SingleConstantPartialRedux<XprType, Eigen::internal::member_redux<std::plus<scalar_type_of_t<XprType>>, scalar_type_of_t<XprType>>> {};


  ///////////
  //  min  //
  ///////////

  template<typename XprType, typename Scalar1, typename Scalar2, typename S>
  struct SingleConstantPartialRedux<XprType, Eigen::internal::member_redux<Eigen::internal::scalar_min_op<Scalar1, Scalar2>, S>>
  {
    struct Op
    {
      template<typename Scalar>
      constexpr Scalar operator()(Scalar x, std::size_t dim) const noexcept
      {
        return x > 0 and dim > 1 ? 0 : x;
      }
    };

    template<typename C, typename Dim, typename Factor>
    static constexpr auto get_constant(const C& c, const Dim& dim, const Factor&) noexcept
    {
      if constexpr (zero_matrix<XprType>)
      {
        using Scalar = std::decay_t<decltype(get_scalar_constant_value(c))>;
        return internal::ScalarConstant<Likelihood::definitely, Scalar, 0>{};
      }
      else if constexpr (detail::is_diag_v<XprType>)
      {
        if constexpr (scalar_constant<C, CompileTimeStatus::known> and not one_by_one_matrix<XprType, Likelihood::maybe>)
          return internal::scalar_constant_operation {Op{}, c, std::integral_constant<std::size_t, 2>{}}; // 2 is an arbitrary number > 1
        else
          return internal::scalar_constant_operation {Op{}, c, dim};
      }
      else return c;
    }
  };


  template<typename XprType, typename...Args>
  struct SingleConstantPartialRedux<XprType, Eigen::internal::member_minCoeff<Args...>>
    : SingleConstantPartialRedux<XprType, Eigen::internal::member_redux<
      Eigen::internal::scalar_min_op<scalar_type_of_t<XprType>, scalar_type_of_t<XprType>>, scalar_type_of_t<XprType>>> {};


  ///////////
  //  max  //
  ///////////

  template<typename XprType, typename Scalar1, typename Scalar2, typename S>
  struct SingleConstantPartialRedux<XprType, Eigen::internal::member_redux<Eigen::internal::scalar_max_op<Scalar1, Scalar2>, S>>
  {
    struct Op
    {
      template<typename Scalar>
      constexpr Scalar operator()(Scalar x, std::size_t dim) const noexcept
      {
        return x < 0 and dim > 1 ? 0 : x;
      }
    };

    template<typename C, typename Dim, typename Factor>
    static constexpr auto get_constant(const C& c, const Dim& dim, const Factor&) noexcept
    {
      if constexpr (zero_matrix<XprType>)
      {
        using Scalar = std::decay_t<decltype(get_scalar_constant_value(c))>;
        return internal::ScalarConstant<Likelihood::definitely, Scalar, 0>{};
      }
      else if constexpr (detail::is_diag_v<XprType>)
      {
        if constexpr (scalar_constant<C, CompileTimeStatus::known> and not one_by_one_matrix<XprType, Likelihood::maybe>)
          return internal::scalar_constant_operation {Op{}, c, std::integral_constant<std::size_t, 2>{}}; // 2 is an arbitrary number > 1
        else
          return internal::scalar_constant_operation {Op{}, c, dim};
      }
      else return c;
    }
  };


  template<typename XprType, typename...Args>
  struct SingleConstantPartialRedux<XprType, Eigen::internal::member_maxCoeff<Args...>>
    : SingleConstantPartialRedux<XprType, Eigen::internal::member_redux<
      Eigen::internal::scalar_max_op<scalar_type_of_t<XprType>, scalar_type_of_t<XprType>>, scalar_type_of_t<XprType>>> {};


  ///////////
  //  and  //
  ///////////

  template<typename XprType, typename S>
  struct SingleConstantPartialRedux<XprType, Eigen::internal::member_redux<std::logical_and<bool>, S>>
  {
    struct Op
    {
      template<typename Scalar>
      constexpr bool operator()(Scalar x, std::size_t dim) const noexcept
      {
        return dim > 1 ? false : static_cast<bool>(x);
      }

      template<typename Scalar>
      constexpr bool operator()(Scalar x) const noexcept { return static_cast<bool>(x); }
    };

    template<typename C, typename Dim, typename Factor>
    static constexpr auto get_constant(const C& c, const Dim& dim, const Factor&) noexcept
    {
      if constexpr (zero_matrix<XprType> or (detail::is_diag_v<XprType> and not one_by_one_matrix<XprType, Likelihood::maybe>))
        return std::false_type{};
      else if constexpr (detail::is_diag_v<XprType>) return internal::scalar_constant_operation {Op{}, c, dim};
      else return internal::scalar_constant_operation {Op{}, c};
    }
  };


  template<typename XprType, typename...Args>
  struct SingleConstantPartialRedux<XprType, Eigen::internal::member_all<Args...>>
    : SingleConstantPartialRedux<XprType, Eigen::internal::member_redux<std::logical_and<bool>, scalar_type_of_t<XprType>>> {};


  template<typename XprType, typename S>
  struct SingleConstantPartialRedux<XprType, Eigen::internal::member_redux<Eigen::internal::scalar_boolean_and_op, S>>
    : SingleConstantPartialRedux<XprType, Eigen::internal::member_redux<std::logical_and<bool>, scalar_type_of_t<XprType>>> {};


  //////////
  //  or  //
  //////////

  template<typename XprType, typename S>
  struct SingleConstantPartialRedux<XprType, Eigen::internal::member_redux<std::logical_or<bool>, S>>
  {
    struct Op
    {
      template<typename Scalar>
      constexpr bool operator()(Scalar x, std::size_t dim) const noexcept
      {
        return dim == 0 ? false : static_cast<bool>(x);
      }
    };

    template<typename C, typename Dim, typename Factor>
    static constexpr auto get_constant(const C& c, const Dim& dim, const Factor& factor) noexcept
    {
      if constexpr (zero_matrix<XprType>) return std::false_type{};
      else return internal::scalar_constant_operation {Op{}, c, dim * factor};
    }
  };


  template<typename XprType, typename...Args>
  struct SingleConstantPartialRedux<XprType, Eigen::internal::member_any<Args...>>
    : SingleConstantPartialRedux<XprType, Eigen::internal::member_redux<std::logical_or<bool>, scalar_type_of_t<XprType>>> {};


  template<typename XprType, typename S>
  struct SingleConstantPartialRedux<XprType, Eigen::internal::member_redux<Eigen::internal::scalar_boolean_or_op, S>>
    : SingleConstantPartialRedux<XprType, Eigen::internal::member_redux<std::logical_or<bool>, scalar_type_of_t<XprType>>> {};


  /////////////
  //  count  //
  /////////////

  template<typename XprType, typename...Args>
  struct SingleConstantPartialRedux<XprType, Eigen::internal::member_count<Args...>>
  {
    struct Op
    {
      template<typename Scalar>
      constexpr Eigen::Index operator()(Scalar x, std::size_t dim) const noexcept
      {
        return static_cast<bool>(x) ? dim : 0;
      }
    };

    template<typename C, typename Dim, typename Factor>
    static constexpr auto get_constant(const C& c, const Dim& dim, const Factor& factor) noexcept
    {
      if constexpr (zero_matrix<XprType>) return std::integral_constant<Eigen::Index, 0>{};
      else if constexpr (detail::is_diag_v<XprType>) return internal::scalar_constant_operation {Op{}, c, factor};
      else return internal::scalar_constant_operation {Op{}, c, dim * factor};
    }
  };


  ///////////////
  //  product  //
  ///////////////

  template<typename XprType, typename T, typename S>
  struct SingleConstantPartialRedux<XprType, Eigen::internal::member_redux<std::multiplies<T>, S>>
  {
    struct Op
    {
      template<typename Scalar>
      constexpr Scalar operator()(Scalar x, std::size_t dim) const noexcept
      {
        return dim == 1 ? x : 0;
      }
    };

    template<typename C, typename Dim, typename Factor>
    static constexpr auto get_constant(const C& c, const Dim& dim, const Factor& factor) noexcept
    {
      using Scalar = std::decay_t<decltype(get_scalar_constant_value(c))>;

      if constexpr (zero_matrix<XprType> or (detail::is_diag_v<XprType> and not one_by_one_matrix<XprType, Likelihood::maybe>))
        return internal::ScalarConstant<Likelihood::definitely, Scalar, 0>{};
      else if constexpr (detail::is_diag_v<XprType>)
        return internal::constexpr_pow(internal::scalar_constant_operation {Op{}, c, dim}, factor);
      else
        return internal::constexpr_pow(c, dim * factor);
    }
  };


  template<typename XprType, typename...Args>
  struct SingleConstantPartialRedux<XprType, Eigen::internal::member_prod<Args...>>
    : SingleConstantPartialRedux<XprType, Eigen::internal::member_redux<std::multiplies<scalar_type_of_t<XprType>>, scalar_type_of_t<XprType>>> {};


  template<typename XprType, typename Scalar1, typename Scalar2, typename S>
  struct SingleConstantPartialRedux<XprType, Eigen::internal::member_redux<Eigen::internal::scalar_product_op<Scalar1, Scalar2>, S>>
    : SingleConstantPartialRedux<XprType, Eigen::internal::member_redux<std::multiplies<scalar_type_of_t<XprType>>, scalar_type_of_t<XprType>>> {};

} // namespace OpenKalman::Eigen3

#endif //OPENKALMAN_EIGEN_TRAITS_FUNCTORS_REDUX_HPP
