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
 * \brief Type traits as applied to native Eigen3 types.
 */

#ifndef OPENKALMAN_EIGEN3_PARTIALREDUXEXPR_HPP
#define OPENKALMAN_EIGEN3_PARTIALREDUXEXPR_HPP

#include <type_traits>


namespace OpenKalman::interface
{

#ifndef __cpp_concepts
  template<typename MatrixType, typename MemberOp, int Direction>
  struct IndexTraits<Eigen::PartialReduxExpr<MatrixType, MemberOp, Direction>>
    : detail::IndexTraits_Eigen_default<Eigen::PartialReduxExpr<MatrixType, MemberOp, Direction>> {};
#endif


  template<typename MatrixType, typename MemberOp, int Direction>
  struct Dependencies<Eigen::PartialReduxExpr<MatrixType, MemberOp, Direction>>
  {
    static constexpr bool has_runtime_parameters = false;

    using type = std::tuple<typename MatrixType::Nested, const MemberOp>;

    template<std::size_t i, typename Arg>
    static decltype(auto) get_nested_matrix(Arg&& arg)
    {
      if constexpr (i == 0)
        return std::forward<Arg>(arg).nestedExpression();
      else
        return std::forward<Arg>(arg).functor();
      static_assert(i <= 1);
    }

    // If a partial redux expression needs to be partially evaluated, it's probably faster to do a full evaluation.
    // Thus, we omit the conversion function.
  };


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
        else if constexpr (is_diag_v<XprType>) return scalar_constant_operation {Op{}, c, factor};
        else return scalar_constant_operation {Op{}, c, scalar_constant_operation {std::multiplies<>{}, dim, factor}};
      }
    };


    template<typename XprType, typename...Args>
    struct SingleConstantPartialRedux<XprType, Eigen::internal::member_stableNorm<Args...>>
    {
      struct Op
      {
        template<typename Scalar>
        constexpr Scalar operator()(Scalar x, std::size_t dim) const noexcept
        {
          auto arg = internal::constexpr_abs(x);
          return internal::constexpr_sqrt(static_cast<Scalar>(dim)) * arg;
        }
      };

      template<typename C, typename Dim, typename Factor>
      static constexpr auto get_constant(const C& c, const Dim& dim, const Factor& factor) noexcept
      {
        using Scalar = std::decay_t<decltype(get_scalar_constant_value(c))>;
        if constexpr (zero_matrix<XprType>) return internal::ScalarConstant<Likelihood::definitely, Scalar, 0>{};
        else if constexpr (is_diag_v<XprType>) return scalar_constant_operation {Op{}, c, factor};
        else return scalar_constant_operation {Op{}, c, scalar_constant_operation {std::multiplies<>{}, dim, factor}};
      }
    };


    template<typename XprType, typename...Args>
    struct SingleConstantPartialRedux<XprType, Eigen::internal::member_hypotNorm<Args...>>
    {
      struct Op
      {
        template<typename Scalar>
        constexpr Scalar operator()(Scalar x, std::size_t dim) const noexcept
        {
          auto arg = internal::constexpr_abs(x);
          return internal::constexpr_sqrt(static_cast<Scalar>(dim)) * arg;
        }
      };

      template<typename C, typename Dim, typename Factor>
      static constexpr auto get_constant(const C& c, const Dim& dim, const Factor& factor) noexcept
      {
        using Scalar = std::decay_t<decltype(get_scalar_constant_value(c))>;
        if constexpr (zero_matrix<XprType>) return internal::ScalarConstant<Likelihood::definitely, Scalar, 0>{};
        else if constexpr (is_diag_v<XprType>) return scalar_constant_operation {Op{}, c, factor};
        else return scalar_constant_operation {Op{}, c, scalar_constant_operation {std::multiplies<>{}, dim, factor}};
      }
    };


    template<typename XprType, typename...Args>
    struct SingleConstantPartialRedux<XprType, Eigen::internal::member_sum<Args...>>
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
        else if constexpr (is_diag_v<XprType>) return scalar_constant_operation {Op{}, c, factor};
        else return scalar_constant_operation {Op{}, c, scalar_constant_operation {std::multiplies<>{}, dim, factor}};
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
            if constexpr (is_diag_v<XprType>) return r * r + i * i;
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
        else if constexpr (is_diag_v<XprType>) return scalar_constant_operation {Op{}, c, factor};
        else return scalar_constant_operation {Op{}, c, scalar_constant_operation {std::multiplies<>{}, dim, factor}};
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
            if constexpr (is_diag_v<XprType>) return r * r + i * i;
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
        else if constexpr (is_diag_v<XprType>) return scalar_constant_operation {Op{}, c, factor};
        else return scalar_constant_operation {Op{}, c, scalar_constant_operation {std::multiplies<>{}, dim, factor}};
      }
    };

    template<typename XprType, typename...Args>
    struct SingleConstantPartialRedux<Eigen::internal::member_mean<Args...>>
    {
      template<typename O>
      struct Op
      {
        template<typename Scalar>
        constexpr Scalar operator()(Scalar x, std::size_t dim) const noexcept { return O{}(x, static_cast<Scalar>(dim)); }
      };

      template<typename C, typename Dim, typename Factor>
      static constexpr auto get_constant(const C& c, const Dim& dim, const Factor& factor) noexcept
      {
        using Scalar = std::decay_t<decltype(get_scalar_constant_value(c))>;
        if constexpr (zero_matrix<XprType>) return internal::ScalarConstant<Likelihood::definitely, Scalar, 0>{};
        else if constexpr (not is_diag_v<XprType>) return c
        else return scalar_constant_operation {Op<std::divides<>>{},
          scalar_constant_operation {Op<std::multiplies<>>{}, c, factor}, dim};
      }
    };
# endif


    template<typename XprType, typename...Args>
    struct SingleConstantPartialRedux<XprType, Eigen::internal::member_minCoeff<Args...>>
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
      static constexpr decltype(auto) get_constant(const C& c, const Dim& dim, const Factor&) noexcept
      {
        if constexpr (zero_matrix<XprType>)
        {
          using Scalar = std::decay_t<decltype(get_scalar_constant_value(c))>;
          return internal::ScalarConstant<Likelihood::definitely, Scalar, 0>{};
        }
        else if constexpr (is_diag_v<XprType>)
        {
          if constexpr (scalar_constant<C, CompileTimeStatus::known> and not one_by_one_matrix<XprType, Likelihood::maybe>)
            return scalar_constant_operation {Op{}, c, std::integral_constant<std::size_t, 2>{}}; // 2 is an arbitrary number > 1
          else
            return scalar_constant_operation {Op{}, c, dim};
        }
        else return c;
      }
    };


    template<typename XprType, typename...Args>
    struct SingleConstantPartialRedux<XprType, Eigen::internal::member_maxCoeff<Args...>>
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
      static constexpr decltype(auto) get_constant(const C& c, const Dim& dim, const Factor&) noexcept
      {
        if constexpr (zero_matrix<XprType>)
        {
          using Scalar = std::decay_t<decltype(get_scalar_constant_value(c))>;
          return internal::ScalarConstant<Likelihood::definitely, Scalar, 0>{};
        }
        else if constexpr (is_diag_v<XprType>)
        {
          if constexpr (scalar_constant<C, CompileTimeStatus::known> and not one_by_one_matrix<XprType, Likelihood::maybe>)
            return scalar_constant_operation {Op{}, c, std::integral_constant<std::size_t, 2>{}}; // 2 is an arbitrary number > 1
          else
            return scalar_constant_operation {Op{}, c, dim};
        }
        else return c;
      }
    };


    template<typename XprType, typename...Args>
    struct SingleConstantPartialRedux<XprType, Eigen::internal::member_all<Args...>>
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
        if constexpr (zero_matrix<XprType> or (is_diag_v<XprType> and not one_by_one_matrix<XprType, Likelihood::maybe>))
          return std::false_type{};
        else if constexpr (is_diag_v<XprType>) return scalar_constant_operation {Op{}, c, dim};
        else return scalar_constant_operation {Op{}, c};
      }
    };


    template<typename XprType, typename...Args>
    struct SingleConstantPartialRedux<XprType, Eigen::internal::member_any<Args...>>
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
        else return scalar_constant_operation {Op{}, c, scalar_constant_operation {std::multiplies<>{}, dim, factor}};
      }
    };


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
        else if constexpr (is_diag_v<XprType>) return scalar_constant_operation {Op{}, c, factor};
        else return scalar_constant_operation {Op{}, c, scalar_constant_operation {std::multiplies<>{}, dim, factor}};
      }
    };


    template<typename XprType, typename...Args>
    struct SingleConstantPartialRedux<XprType, Eigen::internal::member_prod<Args...>>
    {
      struct Op
      {
        template<typename Scalar>
        constexpr Scalar operator()(Scalar x, std::size_t dim) const noexcept
        {
          return dim == 1 ? x : 0;
        }
      };

      struct PowOp
      {
        template<typename Scalar>
        constexpr auto operator()(Scalar x, std::size_t dim) const noexcept
        {
          return OpenKalman::internal::constexpr_pow(x, dim);
        }
      };

      template<typename C, typename Dim, typename Factor>
      static constexpr auto get_constant(const C& c, const Dim& dim, const Factor& factor) noexcept
      {
        using Scalar = std::decay_t<decltype(get_scalar_constant_value(c))>;

        if constexpr (zero_matrix<XprType> or (is_diag_v<XprType> and not one_by_one_matrix<XprType, Likelihood::maybe>))
        {
          return internal::ScalarConstant<Likelihood::definitely, Scalar, 0>{};
        }
        else if constexpr (is_diag_v<XprType>)
        {
          auto c2 = scalar_constant_operation {Op{}, c, dim};
          if constexpr (scalar_constant<decltype(c2), CompileTimeStatus::known>)
          {
            if constexpr (get_scalar_constant_value(c2) == 0) return c2;
            else return scalar_constant_operation {PowOp{}, c2, factor};
          }
          else return scalar_constant_operation {PowOp{}, c2, factor};
        }
        else return scalar_constant_operation {PowOp{}, c, scalar_constant_operation {std::multiplies<>{}, dim, factor}};
      }
    };


    template<typename XprType, int Direction>
    struct is_EigenReplicate : std::false_type {};

    template<typename MatrixType, int RowFactor, int ColFactor, int Direction>
    struct is_EigenReplicate<Eigen::Replicate<MatrixType, RowFactor, ColFactor>, Direction> : std::true_type
    {
    private:
      static constexpr int Efactor = Direction == Eigen::Horizontal ? ColFactor : RowFactor;
    public:
      static constexpr std::size_t direction = Direction == Eigen::Horizontal ? 1 : 0;
      static constexpr std::size_t factor = Efactor == Eigen::Dynamic ? dynamic_size : Efactor;
      static constexpr auto get_nested(const Eigen::Replicate<MatrixType, RowFactor, ColFactor>& xpr) { return xpr.nestedExpression(); }
    };

    template<typename XprType, int Direction>
    struct is_EigenReplicate<const XprType, Direction> : is_EigenReplicate<XprType, Direction> {};


    template<typename MemberOp, std::size_t direction, std::size_t factor,
      typename XprType, typename C, typename Dim>
    constexpr auto get_PartialReduxExpr_replicate(const XprType& xpr, const C& c, const Dim& dim)
    {
      if constexpr (factor == dynamic_size)
      {
        auto d = [](const auto& xpr){
          if constexpr (dynamic_dimension<XprType, direction>) return get_index_dimension_of<direction>(xpr);
          else return std::integral_constant<std::size_t, index_dimension_of_v<XprType, direction>>{};
        }(xpr);
        auto f = get_scalar_constant_value(dim) / d;
        return SingleConstantPartialRedux<XprType, MemberOp>::get_constant(c, d, f);
      }
      else
      {
        std::integral_constant<std::size_t, factor> f;
        scalar_constant_operation d {std::divides<>{}, dim, f};
        return SingleConstantPartialRedux<XprType, MemberOp>::get_constant(c, d, f);
      }
    }


#ifdef __cpp_concepts
    template<typename MemberOp, int Direction, typename XprType, typename Dim>
#else
    template<typename MemberOp, int Direction, typename XprType, typename Dim, std::enable_if_t<
      not is_EigenReplicate<XprType, Direction>::value and
      not eigen_MatrixWrapper<XprType> and not eigen_ArrayWrapper<XprType> and not eigen_wrapper<XprType>, int> = 0>
#endif
    constexpr auto get_PartialReduxExpr_constant(const XprType& xpr, const Dim& dim)
    {
      std::conditional_t<is_diag_v<XprType>, constant_diagonal_coefficient<XprType>, constant_coefficient<XprType>> c {xpr};
      std::integral_constant<std::size_t, 1> f;
      return SingleConstantPartialRedux<XprType, MemberOp>::get_constant(c, dim, f);
    }


#ifdef __cpp_concepts
    template<typename MemberOp, int Direction, typename XprType, typename Dim>
      requires eigen_MatrixWrapper<XprType> or eigen_ArrayWrapper<XprType> or eigen_wrapper<XprType>
#else
    template<typename MemberOp, int Direction, typename XprType, typename Dim, std::enable_if_t<
      not is_EigenReplicate<XprType, Direction>::value and
      (eigen_MatrixWrapper<XprType> or eigen_ArrayWrapper<XprType> or eigen_wrapper<XprType>), int> = 0>
#endif
    constexpr auto get_PartialReduxExpr_constant(const XprType& xpr, const Dim& dim)
    {
      return get_PartialReduxExpr_constant<MemberOp, Direction>(nested_matrix(xpr), dim);
    }


#ifdef __cpp_concepts
    template<typename MemberOp, int Direction, typename XprType, typename Dim>
      requires is_EigenReplicate<XprType, Direction>::value
#else
    template<typename MemberOp, int Direction, typename XprType, typename Dim, std::enable_if_t<
      is_EigenReplicate<XprType, Direction>::value, int> = 0>
#endif
    constexpr auto get_PartialReduxExpr_constant(const XprType& xpr, const Dim& dim)
    {
      constexpr std::size_t direction = is_EigenReplicate<XprType, Direction>::direction;
      constexpr std::size_t factor = is_EigenReplicate<XprType, Direction>::factor;
      decltype(auto) n_xpr = is_EigenReplicate<XprType, Direction>::get_nested(xpr);
      using NXprType = std::decay_t<decltype(n_xpr)>;
      std::conditional_t<is_diag_v<NXprType>, constant_diagonal_coefficient<NXprType>, constant_coefficient<NXprType>> c {n_xpr};
      return get_PartialReduxExpr_replicate<MemberOp, direction, factor>(n_xpr, c, dim);
    }


#ifdef __cpp_concepts
    template<typename MemberOp, int Direction, typename UnaryOp, typename XprType, typename Dim>
      requires is_EigenReplicate<XprType, Direction>::value
#else
    template<typename MemberOp, int Direction, typename UnaryOp, typename XprType, typename Dim, std::enable_if_t<
      is_EigenReplicate<XprType, Direction>::value, int> = 0>
#endif
    constexpr auto get_PartialReduxExpr_constant(const Eigen::CwiseUnaryOp<UnaryOp, XprType>& xpr, const Dim& dim)
    {
      constexpr std::size_t direction = is_EigenReplicate<XprType, Direction>::direction;
      constexpr std::size_t factor = is_EigenReplicate<XprType, Direction>::factor;
      decltype(auto) n_xpr = is_EigenReplicate<XprType, Direction>::get_nested(xpr.nestedExpression());
      using NXprType = std::decay_t<decltype(n_xpr)>;
      auto uop = Eigen::CwiseUnaryOp<UnaryOp, NXprType> {n_xpr};
      auto c = Eigen3::FunctorTraits<UnaryOp, NXprType>::template get_constant<is_diag_v<XprType>>(uop);
      return get_PartialReduxExpr_replicate<MemberOp, direction, factor>(n_xpr, c, dim);
    }


#ifdef __cpp_concepts
    template<typename MemberOp, int Direction, typename ViewOp, typename XprType, typename Dim>
      requires is_EigenReplicate<XprType, Direction>::value
#else
    template<typename MemberOp, int Direction, typename ViewOp, typename XprType, typename Dim, std::enable_if_t<
      is_EigenReplicate<XprType, Direction>::value, int> = 0>
#endif
    constexpr auto get_PartialReduxExpr_constant(const Eigen::CwiseUnaryView<ViewOp, XprType>& xpr, const Dim& dim)
    {
      constexpr std::size_t direction = is_EigenReplicate<XprType, Direction>::direction;
      constexpr std::size_t factor = is_EigenReplicate<XprType, Direction>::factor;
      decltype(auto) n_xpr = is_EigenReplicate<XprType, Direction>::get_nested(xpr.nestedExpression());
      using NXprType = std::decay_t<decltype(n_xpr)>;
      auto uop = Eigen::CwiseUnaryOp<ViewOp, const NXprType> {n_xpr};
      auto c = Eigen3::FunctorTraits<ViewOp, NXprType>::template get_constant<is_diag_v<XprType>>(uop);
      return get_PartialReduxExpr_replicate<MemberOp, direction, factor>(n_xpr, c, dim);
    }

  } // namespace detail


  template<typename MatrixType, typename MemberOp, int Direction>
  struct SingleConstant<Eigen::PartialReduxExpr<MatrixType, MemberOp, Direction>>
  {
    const Eigen::PartialReduxExpr<MatrixType, MemberOp, Direction>& xpr;

    constexpr auto get_constant()
    {
      // colwise (acting on columns) is Eigen::Vertical and rowwise (acting on rows) is Eigen::Horizontal
      constexpr std::size_t N = Direction == Eigen::Horizontal ? 1 : 0;
      const auto& x {xpr.nestedExpression()};
      auto dim = [](const auto& x){
        if constexpr (dynamic_dimension<MatrixType, N>) return get_index_dimension_of<N>(x);
        else return std::integral_constant<std::size_t, index_dimension_of_v<MatrixType, N>>{};
      }(x);

      return detail::get_PartialReduxExpr_constant<MemberOp, Direction>(x, dim);
    }
  };

} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN3_PARTIALREDUXEXPR_HPP
